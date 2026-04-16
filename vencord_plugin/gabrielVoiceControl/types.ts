/*
 * Gabriel Voice Control - Vencord UserPlugin
 * Copyright (c) 2025 HoppouAI
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

export interface WsCommand {
    op: string;
    nonce?: string;
    [key: string]: any;
}

export interface WsResponse {
    op: string;
    nonce?: string;
    success: boolean;
    data?: any;
    error?: string;
}

// Commands the AI can send
export type CommandOp =
    | "join_voice"      // Join a voice channel (server or DM)
    | "leave_voice"     // Leave current voice channel
    | "call_user"       // Ring a user in DM
    | "answer_call"     // Accept an incoming call
    | "hang_up"         // End current call
    | "get_voice_state" // Get current voice connection state
    | "set_mute"        // Mute/unmute self
    | "set_deaf"        // Deafen/undeafen self
    | "ping";           // Health check

// Events the plugin can push to the AI
export type EventOp =
    | "voice_state_update"  // Voice state changed
    | "call_incoming"       // Someone is calling
    | "call_ended"          // Call ended
    | "user_joined_voice"   // Someone joined the voice channel
    | "user_left_voice";    // Someone left the voice channel
