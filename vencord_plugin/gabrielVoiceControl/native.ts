/*
 * Gabriel Voice Control - Native (Node.js) Module
 * Runs in Electron's main process via Vencord's native IPC bridge.
 * AI connects via WS, native queues commands, renderer polls and executes.
 * Copyright (c) 2025 HoppouAI
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { createServer } from "http";
import { createHash } from "crypto";

// --- Minimal WebSocket implementation using Node.js built-ins ---

const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const OPEN = 1;
const CLOSING = 2;
const CLOSED = 3;

interface WSClient {
    socket: any;
    state: number;
}

function acceptKey(key: string): string {
    return createHash("sha1").update(key + WS_GUID).digest("base64");
}

function encodeFrame(data: string): Buffer {
    const payload = Buffer.from(data, "utf-8");
    const len = payload.length;
    let header: Buffer;

    if (len < 126) {
        header = Buffer.alloc(2);
        header[0] = 0x81;
        header[1] = len;
    } else if (len < 65536) {
        header = Buffer.alloc(4);
        header[0] = 0x81;
        header[1] = 126;
        header.writeUInt16BE(len, 2);
    } else {
        header = Buffer.alloc(10);
        header[0] = 0x81;
        header[1] = 127;
        header.writeBigUInt64BE(BigInt(len), 2);
    }

    return Buffer.concat([header, payload]);
}

function wsSend(client: WSClient, data: string) {
    if (client.state === OPEN && !client.socket.destroyed) {
        client.socket.write(encodeFrame(data));
    }
}

function wsClose(client: WSClient, code: number, reason: string) {
    if (client.state === OPEN) {
        client.state = CLOSING;
        const buf = Buffer.alloc(2 + Buffer.byteLength(reason, "utf-8"));
        buf.writeUInt16BE(code, 0);
        buf.write(reason, 2, "utf-8");
        const frame = Buffer.alloc(2);
        frame[0] = 0x88;
        frame[1] = buf.length;
        client.socket.write(Buffer.concat([frame, buf]));
        client.socket.end();
        client.state = CLOSED;
    }
}

// --- Server state ---

let httpServer: any = null;
const clients = new Set<WSClient>();

interface PendingCommand {
    id: string;
    cmd: any;
    client: WSClient;
}
const pendingCommands = new Map<string, PendingCommand>();
let cmdCounter = 0;

function genId(): string {
    return `cmd_${++cmdCounter}_${Date.now()}`;
}

function removeClient(client: WSClient) {
    clients.delete(client);
    for (const [id, pending] of pendingCommands) {
        if (pending.client === client) pendingCommands.delete(id);
    }
}

function parseFrames(buf: Buffer, callback: (opcode: number, payload: Buffer) => void): number {
    let offset = 0;
    while (offset < buf.length) {
        if (buf.length - offset < 2) break;

        const byte0 = buf[offset];
        const byte1 = buf[offset + 1];
        const opcode = byte0 & 0x0f;
        const masked = (byte1 & 0x80) !== 0;
        let payloadLen = byte1 & 0x7f;
        let headerLen = 2;

        if (payloadLen === 126) {
            if (buf.length - offset < 4) break;
            payloadLen = buf.readUInt16BE(offset + 2);
            headerLen = 4;
        } else if (payloadLen === 127) {
            if (buf.length - offset < 10) break;
            payloadLen = Number(buf.readBigUInt64BE(offset + 2));
            headerLen = 10;
        }

        if (masked) headerLen += 4;
        if (buf.length - offset < headerLen + payloadLen) break;

        let payload = buf.subarray(offset + headerLen, offset + headerLen + payloadLen);
        if (masked) {
            const maskKey = buf.subarray(offset + headerLen - 4, offset + headerLen);
            payload = Buffer.from(payload);
            for (let i = 0; i < payload.length; i++) {
                payload[i] ^= maskKey[i % 4];
            }
        }

        callback(opcode, payload);
        offset += headerLen + payloadLen;
    }
    return offset;
}

// --- IPC exports (called from renderer via VencordNative.pluginHelpers) ---
// Electron's ipcMain.handle passes (event, ...args) so all exports receive _event first

export function startServer(_event: any, port: number) {
    if (httpServer) return { success: true, message: "Already running" };

    try {
        httpServer = createServer((_req: any, res: any) => {
            res.writeHead(426, { "Content-Type": "text/plain" });
            res.end("WebSocket required");
        });

        httpServer.on("upgrade", (req: any, socket: any, head: Buffer) => {
            const ip = socket.remoteAddress;
            if (ip !== "127.0.0.1" && ip !== "::1" && ip !== "::ffff:127.0.0.1") {
                socket.destroy();
                return;
            }

            const key = req.headers["sec-websocket-key"];
            if (!key) {
                socket.destroy();
                return;
            }

            const accept = acceptKey(key);
            socket.write(
                "HTTP/1.1 101 Switching Protocols\r\n" +
                "Upgrade: websocket\r\n" +
                "Connection: Upgrade\r\n" +
                `Sec-WebSocket-Accept: ${accept}\r\n` +
                "\r\n"
            );

            const client: WSClient = { socket, state: OPEN };
            clients.add(client);
            console.log("[GabrielVoice] AI client connected");

            let remainder = Buffer.alloc(0);
            if (head && head.length > 0) remainder = Buffer.from(head);

            socket.on("data", (chunk: Buffer) => {
                remainder = Buffer.concat([remainder, chunk]);
                const consumed = parseFrames(remainder, (opcode, payload) => {
                    if (opcode === 0x08) {
                        client.state = CLOSED;
                        socket.end();
                        removeClient(client);
                        console.log("[GabrielVoice] AI client disconnected");
                        return;
                    }
                    if (opcode === 0x09) {
                        const pong = Buffer.alloc(2 + payload.length);
                        pong[0] = 0x8a;
                        pong[1] = payload.length;
                        payload.copy(pong, 2);
                        socket.write(pong);
                        return;
                    }
                    if (opcode === 0x01) {
                        try {
                            const msg = JSON.parse(payload.toString("utf-8"));
                            if (!msg.op) return;

                            if (msg.op === "ping") {
                                wsSend(client, JSON.stringify({ op: "pong", nonce: msg.nonce, success: true }));
                                return;
                            }

                            const id = genId();
                            pendingCommands.set(id, { id, cmd: msg, client });
                        } catch (e: any) {
                            wsSend(client, JSON.stringify({ op: "error", success: false, error: `Parse error: ${e.message}` }));
                        }
                    }
                });
                remainder = remainder.subarray(consumed);
            });

            socket.on("close", () => {
                removeClient(client);
            });

            socket.on("error", (err: any) => {
                console.error("[GabrielVoice] Socket error:", err.message);
                removeClient(client);
            });
        });

        httpServer.on("error", (err: any) => {
            console.error("[GabrielVoice] Server error:", err.message);
        });

        httpServer.listen(port, "127.0.0.1", () => {
            console.log(`[GabrielVoice] WS server on 127.0.0.1:${port}`);
        });

        return { success: true, port };
    } catch (e: any) {
        return { success: false, error: e.message };
    }
}

export function stopServer(_event: any) {
    if (httpServer) {
        for (const client of clients) wsClose(client, 1000, "Plugin disabled");
        clients.clear();
        pendingCommands.clear();
        httpServer.close();
        httpServer = null;
        console.log("[GabrielVoice] Server stopped");
    }
    return { success: true };
}

export function drainCommands(_event: any) {
    const result: Array<{ id: string; cmd: any }> = [];
    for (const [id, pending] of pendingCommands) {
        result.push({ id, cmd: pending.cmd });
    }
    return result;
}

export function resolveCommand(_event: any, id: string, response: any) {
    const pending = pendingCommands.get(id);
    if (!pending) return { success: false, error: "Command not found or expired" };

    wsSend(pending.client, JSON.stringify({
        op: "result",
        nonce: pending.cmd.nonce,
        ...response,
    }));
    pendingCommands.delete(id);
    return { success: true };
}

export function broadcastEvent(_event: any, eventData: any) {
    const payload = JSON.stringify(eventData);
    let sent = 0;
    for (const client of clients) {
        if (client.state === OPEN) {
            wsSend(client, payload);
            sent++;
        }
    }
    return { success: true, sent };
}

export function getStatus(_event: any) {
    return {
        running: httpServer !== null,
        clients: clients.size,
        pendingCommands: pendingCommands.size,
    };
}
