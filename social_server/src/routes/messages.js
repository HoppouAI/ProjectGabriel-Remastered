const express = require("express");
const db = require("../database");

const router = express.Router();

let maxMessageLength = 2000;

function setMaxMessageLength(len) {
  maxMessageLength = len;
}

// POST /api/messages/send - Send a message
router.post("/messages/send", (req, res) => {
  const { to, content } = req.body || {};

  if (!to || typeof to !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'to' field" });
  }
  if (!content || typeof content !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'content' field" });
  }
  if (content.length > maxMessageLength) {
    return res.status(400).json({ error: `Message too long (max ${maxMessageLength} characters)` });
  }
  if (to === req.username) {
    return res.status(400).json({ error: "Cannot send a message to yourself" });
  }

  // Check target user exists
  const targetUser = db.getUser(to);
  if (!targetUser) {
    return res.status(404).json({ error: `User "${to}" not found` });
  }

  // Check block status
  if (db.isBlocked(req.username, to)) {
    return res.status(403).json({ error: "Cannot send message - blocked" });
  }

  try {
    const result = db.sendMessage(req.username, to, content);
    const messageId = result.lastInsertRowid;

    // Notify via WebSocket if available
    const wsManager = req.app.get("wsManager");
    if (wsManager) {
      wsManager.notifyUser(to, {
        type: "new_message",
        message: {
          id: Number(messageId),
          from_user: req.username,
          to_user: to,
          content: content,
          timestamp: new Date().toISOString(),
          read: 0,
        },
      });
    }

    return res.json({ result: "sent", message_id: Number(messageId) });
  } catch (e) {
    return res.status(500).json({ error: "Failed to send message", details: e.message });
  }
});

// GET /api/messages/recent - Fetch recent messages involving you
router.get("/messages/recent", (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 200);
  try {
    const messages = db.getRecentMessages(req.username, limit);
    return res.json({ messages: formatMessages(messages) });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch messages" });
  }
});

// GET /api/messages/user/:username - Fetch messages with a specific user
router.get("/messages/user/:username", (req, res) => {
  const otherUser = req.params.username;
  if (!otherUser || typeof otherUser !== "string") {
    return res.status(400).json({ error: "Invalid username" });
  }
  const limit = Math.min(parseInt(req.query.limit) || 50, 200);
  try {
    const messages = db.getMessagesByUser(req.username, otherUser, limit);
    return res.json({ messages: formatMessages(messages) });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch messages" });
  }
});

// GET /api/messages/unread - Fetch unread messages
router.get("/messages/unread", (req, res) => {
  try {
    const messages = db.getUnreadMessages(req.username);
    return res.json({ messages: formatMessages(messages), count: messages.length });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch unread messages" });
  }
});

// POST /api/messages/read - Mark messages as read
router.post("/messages/read", (req, res) => {
  const { from } = req.body || {};
  try {
    db.markMessagesRead(req.username, from || null);
    return res.json({ result: "ok" });
  } catch (e) {
    return res.status(500).json({ error: "Failed to mark messages read" });
  }
});

// Format messages with 12h time
function formatMessages(messages) {
  return messages.map((msg) => {
    const date = new Date(msg.timestamp + "Z"); // SQLite stores UTC
    return {
      id: msg.id,
      from: msg.from_user,
      to: msg.to_user,
      content: msg.content,
      time: date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true }),
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
      timestamp: msg.timestamp,
      read: !!msg.read,
    };
  });
}

module.exports = router;
module.exports.setMaxMessageLength = setMaxMessageLength;
