/*
 * Gabriel Voice Control - Vencord UserPlugin
 * Exposes voice channel / call control to ProjectGabriel AI via WebSocket.
 * Copyright (c) 2025 HoppouAI
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import definePlugin, { OptionType } from "@utils/types";
import { findByProps, findStore } from "@webpack";
import { FluxDispatcher, RestAPI } from "@webpack/common";

// IPC helpers for communicating with native.ts (resolved lazily)
function getNative() {
    return VencordNative.pluginHelpers.GabrielVoiceControl as {
        startServer(port: number): Promise<any>;
        stopServer(): Promise<any>;
        drainCommands(): Promise<Array<{ id: string; cmd: any }>>;
        resolveCommand(id: string, response: any): Promise<any>;
        broadcastEvent(event: any): Promise<any>;
        getStatus(): Promise<any>;
    } | undefined;
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

// --- Helpers (DM channel creation + call ring/stop via Discord REST API) ---

async function getOrCreateDMChannel(userId: string): Promise<string | null> {
    const ChannelStore = findStore("ChannelStore");
    const cached = ChannelStore?.getDMFromUserId?.(userId);
    if (cached) return cached;

    try {
        const resp = await RestAPI.post({
            url: "/users/@me/channels",
            body: { recipients: [userId] },
        });
        return resp?.body?.id || null;
    } catch {
        return null;
    }
}

async function ringChannel(channelId: string, recipients?: string[]): Promise<boolean> {
    try {
        await RestAPI.post({
            url: `/channels/${channelId}/call/ring`,
            body: { recipients: recipients || null },
        });
        return true;
    } catch {
        return false;
    }
}

async function stopRinging(channelId: string): Promise<boolean> {
    try {
        await RestAPI.post({
            url: `/channels/${channelId}/call/stop-ringing`,
            body: {},
        });
        return true;
    } catch {
        return false;
    }
}

// --- Command Handlers (use lazy store lookups to avoid module-eval crashes) ---

async function handleJoinVoice(args: any) {
    const channelId = args.channel_id;
    if (!channelId) return { success: false, error: "channel_id required" };

    const ChannelStore = findStore("ChannelStore");
    const channel = ChannelStore?.getChannel(channelId);
    if (!channel) return { success: false, error: "Channel not found" };

    const guildId = channel.guild_id || null;

    try {
        const VoiceActions = findByProps("selectVoiceChannel");
        await VoiceActions.selectVoiceChannel(channelId);
        return { success: true, data: { channel_id: channelId, guild_id: guildId } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleLeaveVoice() {
    try {
        const VoiceActions = findByProps("selectVoiceChannel");
        await VoiceActions.selectVoiceChannel(null);
        return { success: true };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleCallUser(args: any) {
    const channelId = args.channel_id;
    if (!channelId) return { success: false, error: "channel_id required" };

    try {
        // Join voice in the DM/group DM channel
        const VoiceActions = findByProps("selectVoiceChannel");
        if (!VoiceActions) return { success: false, error: "VoiceActions not found" };
        await VoiceActions.selectVoiceChannel(channelId);

        // Ring the recipients
        const rang = await ringChannel(channelId);
        return { success: true, data: { channel_id: channelId, rang } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleCallUserById(args: any) {
    const userId = args.user_id;
    if (!userId) return { success: false, error: "user_id required" };

    try {
        // Get or create DM channel with the user
        const channelId = await getOrCreateDMChannel(userId);
        if (!channelId) return { success: false, error: "Could not create DM channel" };

        // Join voice in the DM channel
        const VoiceActions = findByProps("selectVoiceChannel");
        if (!VoiceActions) return { success: false, error: "VoiceActions not found" };
        await VoiceActions.selectVoiceChannel(channelId);

        // Ring the specific user
        const rang = await ringChannel(channelId, [userId]);
        return { success: true, data: { channel_id: channelId, user_id: userId, rang } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleAnswerCall(args: any) {
    const channelId = args.channel_id;
    if (!channelId) return { success: false, error: "channel_id required" };

    try {
        const VoiceActions = findByProps("selectVoiceChannel");
        await VoiceActions.selectVoiceChannel(channelId);
        return { success: true, data: { channel_id: channelId } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleHangUp() {
    try {
        // Get current channel before disconnecting (to stop ringing)
        const VoiceStateStore = findStore("VoiceStateStore");
        const UserStore = findStore("UserStore");
        const me = UserStore?.getCurrentUser();
        const myState = me ? VoiceStateStore?.getVoiceStateForUser(me.id) : null;
        const currentChannelId = myState?.channelId;

        // Disconnect from voice
        const VoiceActions = findByProps("selectVoiceChannel");
        await VoiceActions.selectVoiceChannel(null);

        // Stop ringing if we were in a DM/group DM
        if (currentChannelId) {
            await stopRinging(currentChannelId);
        }
        return { success: true };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

function handleGetVoiceState() {
    try {
        const VoiceStateStore = findStore("VoiceStateStore");
        const ChannelStore = findStore("ChannelStore");
        const UserStore = findStore("UserStore");

        const me = UserStore?.getCurrentUser();
        if (!me) return { success: false, error: "Not logged in" };

        const myState = VoiceStateStore?.getVoiceStateForUser(me.id);
        const channelId = myState?.channelId || null;
        const channel = channelId ? ChannelStore?.getChannel(channelId) : null;

        const usersInChannel: Array<{ id: string; name: string; mute: boolean; deaf: boolean }> = [];
        if (channelId) {
            const voiceStates = VoiceStateStore?.getVoiceStatesForChannel(channelId);
            if (voiceStates) {
                for (const [userId, state] of Object.entries(voiceStates) as any) {
                    const user = UserStore?.getUser(userId);
                    usersInChannel.push({
                        id: userId,
                        name: user?.username || "Unknown",
                        mute: state.mute || state.selfMute,
                        deaf: state.deaf || state.selfDeaf,
                    });
                }
            }
        }

        return {
            success: true,
            data: {
                connected: !!channelId,
                channel_id: channelId,
                guild_id: channel?.guild_id || null,
                channel_name: channel?.name || null,
                self_mute: myState?.selfMute || false,
                self_deaf: myState?.selfDeaf || false,
                users: usersInChannel,
            },
        };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

function handleSetMute(args: any) {
    try {
        FluxDispatcher.dispatch({
            type: "AUDIO_TOGGLE_SELF_MUTE",
            context: "default",
            syncRemote: true,
        });
        return { success: true, data: { mute: args.mute } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

function handleSetDeaf(args: any) {
    try {
        FluxDispatcher.dispatch({
            type: "AUDIO_TOGGLE_SELF_DEAF",
            context: "default",
            syncRemote: true,
        });
        return { success: true, data: { deaf: args.deaf } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

// --- Command Router ---

async function processCommand(cmd: any): Promise<any> {
    switch (cmd.op) {
        case "join_voice": return handleJoinVoice(cmd);
        case "leave_voice": return handleLeaveVoice();
        case "call_user": return handleCallUser(cmd);
        case "call_user_by_id": return handleCallUserById(cmd);
        case "answer_call": return handleAnswerCall(cmd);
        case "hang_up": return handleHangUp();
        case "get_voice_state": return handleGetVoiceState();
        case "set_mute": return handleSetMute(cmd);
        case "set_deaf": return handleSetDeaf(cmd);
        default: return { success: false, error: `Unknown op: ${cmd.op}` };
    }
}

// --- Flux Event Listeners ---

function onVoiceStateUpdate(event: any) {
    try {
        getNative()?.broadcastEvent({
            op: "voice_state_update",
            data: {
                user_id: event.userId,
                channel_id: event.channelId,
                guild_id: event.guildId,
            },
        });
    } catch { }
}

function onCallCreate(event: any) {
    try {
        getNative()?.broadcastEvent({
            op: "call_incoming",
            data: {
                channel_id: event.channelId,
                ringing: event.ringing,
            },
        });
    } catch { }
}

// --- Polling Loop ---

async function pollCommands() {
    try {
        const native = getNative();
        if (!native) return;
        const commands = await native.drainCommands();
        for (const { id, cmd } of commands) {
            const result = await processCommand(cmd);
            await native.resolveCommand(id, result);
        }
    } catch { }
}

// --- Plugin Definition ---

export default definePlugin({
    name: "GabrielVoiceControl",
    description: "Exposes Discord voice/call control to ProjectGabriel AI via WebSocket API",
    authors: [{
        name: "HoppouAI",
        id: 0n,
    }],

    settings: {
        port: {
            type: OptionType.NUMBER,
            description: "WebSocket server port (localhost only)",
            default: 9473,
        },
    },

    flux: {
        VOICE_STATE_UPDATES: onVoiceStateUpdate,
        CALL_CREATE: onCallCreate,
    },

    async start() {
        try {
            const native = getNative();
            if (!native) {
                console.error("[GabrielVoice] Native helpers not available");
                return;
            }
            const port = this.settings?.store?.port ?? 9473;
            const result = await native.startServer(port);
            if (!result?.success) {
                console.error("[GabrielVoice] Failed to start server:", result?.error);
                return;
            }
            pollInterval = setInterval(pollCommands, 100);
            console.log("[GabrielVoice] Plugin started on port", port);
        } catch (e) {
            console.error("[GabrielVoice] start() error:", e);
        }
    },

    stop() {
        try {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
            getNative()?.stopServer();
            console.log("[GabrielVoice] Plugin stopped");
        } catch (e) {
            console.error("[GabrielVoice] stop() error:", e);
        }
    },
});
