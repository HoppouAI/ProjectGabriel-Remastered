/*
 * Gabriel Voice Control - Vencord UserPlugin
 * Exposes voice channel / call control to ProjectGabriel AI via WebSocket.
 * Copyright (c) 2025 HoppouAI
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import definePlugin, { OptionType } from "@utils/types";
import { findByProps, findStore } from "@webpack";
import { FluxDispatcher } from "@webpack/common";

// Discord internal stores and modules (found via webpack)
const VoiceStateStore = findStore("VoiceStateStore");
const ChannelStore = findStore("ChannelStore");
const UserStore = findStore("UserStore");

// Voice channel actions
const VoiceActions = findByProps("selectVoiceChannel");
const CallActions = findByProps("startCall", "stopRinging");

// IPC helpers for communicating with native.ts
const Native = VencordNative.pluginHelpers.GabrielVoiceControl as {
    startServer(port: number): Promise<any>;
    stopServer(): Promise<any>;
    drainCommands(): Promise<Array<{ id: string; cmd: any }>>;
    resolveCommand(id: string, response: any): Promise<any>;
    broadcastEvent(event: any): Promise<any>;
    getStatus(): Promise<any>;
};

let pollInterval: ReturnType<typeof setInterval> | null = null;

// --- Command Handlers ---

async function handleJoinVoice(args: any) {
    const channelId = args.channel_id;
    if (!channelId) return { success: false, error: "channel_id required" };

    const channel = ChannelStore.getChannel(channelId);
    if (!channel) return { success: false, error: "Channel not found" };

    const guildId = channel.guild_id || null;

    try {
        await VoiceActions.selectVoiceChannel(channelId);
        return { success: true, data: { channel_id: channelId, guild_id: guildId } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleLeaveVoice() {
    try {
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
        await CallActions.startCall(channelId);
        return { success: true, data: { channel_id: channelId } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleAnswerCall(args: any) {
    const channelId = args.channel_id;
    if (!channelId) return { success: false, error: "channel_id required" };

    try {
        // Answering = joining the voice channel of the call
        await VoiceActions.selectVoiceChannel(channelId);
        return { success: true, data: { channel_id: channelId } };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

async function handleHangUp() {
    try {
        await VoiceActions.selectVoiceChannel(null);
        return { success: true };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

function handleGetVoiceState() {
    const me = UserStore.getCurrentUser();
    if (!me) return { success: false, error: "Not logged in" };

    const voiceStates = VoiceStateStore.getVoiceStatesForChannel(
        VoiceStateStore.getVoiceStateForUser(me.id)?.channelId
    );

    const myState = VoiceStateStore.getVoiceStateForUser(me.id);
    const channelId = myState?.channelId || null;
    const channel = channelId ? ChannelStore.getChannel(channelId) : null;

    const usersInChannel: Array<{ id: string; name: string; mute: boolean; deaf: boolean }> = [];
    if (voiceStates) {
        for (const [userId, state] of Object.entries(voiceStates) as any) {
            const user = UserStore.getUser(userId);
            usersInChannel.push({
                id: userId,
                name: user?.username || "Unknown",
                mute: state.mute || state.selfMute,
                deaf: state.deaf || state.selfDeaf,
            });
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
}

function handleSetMute(args: any) {
    const mute = args.mute === true || args.mute === "true";
    FluxDispatcher.dispatch({
        type: "AUDIO_TOGGLE_SELF_MUTE",
        context: "default",
        syncRemote: true,
    });
    return { success: true, data: { mute } };
}

function handleSetDeaf(args: any) {
    const deaf = args.deaf === true || args.deaf === "true";
    FluxDispatcher.dispatch({
        type: "AUDIO_TOGGLE_SELF_DEAF",
        context: "default",
        syncRemote: true,
    });
    return { success: true, data: { deaf } };
}

// --- Command Router ---

async function processCommand(cmd: any): Promise<any> {
    switch (cmd.op) {
        case "join_voice": return handleJoinVoice(cmd);
        case "leave_voice": return handleLeaveVoice();
        case "call_user": return handleCallUser(cmd);
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
    Native.broadcastEvent({
        op: "voice_state_update",
        data: {
            user_id: event.userId,
            channel_id: event.channelId,
            guild_id: event.guildId,
        },
    });
}

function onCallCreate(event: any) {
    Native.broadcastEvent({
        op: "call_incoming",
        data: {
            channel_id: event.channelId,
            ringing: event.ringing,
        },
    });
}

// --- Polling Loop ---

async function pollCommands() {
    try {
        const commands = await Native.drainCommands();
        for (const { id, cmd } of commands) {
            const result = await processCommand(cmd);
            await Native.resolveCommand(id, result);
        }
    } catch (e) {
        // Silently ignore poll errors
    }
}

// --- Plugin Definition ---

export default definePlugin({
    name: "GabrielVoiceControl",
    description: "Exposes Discord voice/call control to ProjectGabriel AI via WebSocket API",
    authors: [{
        name: "HoppouAI",
        id: 0n, // Replace with your Discord user ID as BigInt
    }],

    settings: {
        port: {
            type: OptionType.NUMBER,
            description: "WebSocket server port (localhost only)",
            default: 6463,
        },
    },

    flux: {
        VOICE_STATE_UPDATES: onVoiceStateUpdate,
        CALL_CREATE: onCallCreate,
    },

    async start() {
        try {
            const port = this.settings?.store?.port ?? 6463;
            const result = await Native.startServer(port);
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
            Native.stopServer();
            console.log("[GabrielVoice] Plugin stopped");
        } catch (e) {
            console.error("[GabrielVoice] stop() error:", e);
        }
    },
});
