/*
 * Gabriel Voice Control - Native (Node.js) Module
 * Runs a WebSocket server that ProjectGabriel connects to for voice control.
 * Copyright (c) 2025 HoppouAI
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Architecture:
 *   AI (ProjectGabriel) connects via WS to native.ts, which uses IPC to talk to index.ts (renderer)
 *
 * Commands flow: AI sends WS command, native queues it, renderer polls and executes, result sent back
 * Events flow:  renderer detects Discord event, IPC to native, native broadcasts to AI via WS
 */

import { Server as WebSocketServer, WebSocket } from "ws";
import type { IpcMainInvokeEvent } from "electron";

let wss: WebSocketServer | null = null;
const clients = new Set<WebSocket>();

// Command queue: AI sends commands, renderer polls and processes them
interface PendingCommand {
    id: string;
    cmd: any;
    ws: WebSocket;
    resolve?: (response: any) => void;
}
const pendingCommands = new Map<string, PendingCommand>();
let cmdCounter = 0;

function genId(): string {
    return `cmd_${++cmdCounter}_${Date.now()}`;
}

// --- IPC exports (called from renderer via VencordNative.pluginHelpers) ---

export function startServer(_: IpcMainInvokeEvent, port: number) {
    if (wss) return { success: true, message: "Already running" };

    try {
        wss = new WebSocketServer({ port, host: "127.0.0.1" });

        wss.on("connection", (ws, req) => {
            const ip = req.socket.remoteAddress;
            if (ip !== "127.0.0.1" && ip !== "::1" && ip !== "::ffff:127.0.0.1") {
                ws.close(4003, "Forbidden: localhost only");
                return;
            }

            clients.add(ws);
            console.log("[GabrielVoice] AI client connected");

            ws.on("message", (raw) => {
                try {
                    const msg = JSON.parse(raw.toString());
                    if (!msg.op) return;

                    if (msg.op === "ping") {
                        ws.send(JSON.stringify({ op: "pong", nonce: msg.nonce, success: true }));
                        return;
                    }

                    // Queue command for renderer to process
                    const id = genId();
                    pendingCommands.set(id, { id, cmd: msg, ws });

                } catch (e: any) {
                    ws.send(JSON.stringify({ op: "error", success: false, error: `Parse error: ${e.message}` }));
                }
            });

            ws.on("close", () => {
                clients.delete(ws);
                // Clean up pending commands for this client
                for (const [id, pending] of pendingCommands) {
                    if (pending.ws === ws) pendingCommands.delete(id);
                }
                console.log("[GabrielVoice] AI client disconnected");
            });

            ws.on("error", (err) => {
                console.error("[GabrielVoice] WS error:", err.message);
                clients.delete(ws);
            });
        });

        wss.on("error", (err: any) => {
            console.error("[GabrielVoice] Server error:", err.message);
        });

        console.log(`[GabrielVoice] WS server on 127.0.0.1:${port}`);
        return { success: true, port };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

export function stopServer(_?: IpcMainInvokeEvent) {
    if (wss) {
        for (const client of clients) client.close(1000, "Plugin disabled");
        clients.clear();
        pendingCommands.clear();
        wss.close();
        wss = null;
        console.log("[GabrielVoice] Server stopped");
    }
    return { success: true };
}

/**
 * Renderer polls this to get commands that need processing.
 * Returns array of { id, cmd } objects.
 */
export function drainCommands(_?: IpcMainInvokeEvent) {
    const result: Array<{ id: string; cmd: any }> = [];
    for (const [id, pending] of pendingCommands) {
        result.push({ id, cmd: pending.cmd });
    }
    return result;
}

/**
 * Renderer calls this after processing a command to send the result back to the AI.
 */
export function resolveCommand(_: IpcMainInvokeEvent, id: string, response: any) {
    const pending = pendingCommands.get(id);
    if (!pending) return { success: false, error: "Command not found or expired" };

    if (pending.ws.readyState === WebSocket.OPEN) {
        pending.ws.send(JSON.stringify({
            op: "result",
            nonce: pending.cmd.nonce,
            ...response,
        }));
    }
    pendingCommands.delete(id);
    return { success: true };
}

/**
 * Renderer calls this to broadcast a Discord event to all connected AI clients.
 * Events: voice_state_update, call_incoming, call_ended, user_joined_voice, user_left_voice
 */
export function broadcastEvent(_: IpcMainInvokeEvent, event: any) {
    const payload = JSON.stringify(event);
    let sent = 0;
    for (const client of clients) {
        if (client.readyState === WebSocket.OPEN) {
            client.send(payload);
            sent++;
        }
    }
    return { success: true, sent };
}

export function getStatus(_?: IpcMainInvokeEvent) {
    return {
        running: wss !== null,
        clients: clients.size,
        pendingCommands: pendingCommands.size,
    };
}
