const express = require("express");
const db = require("../database");

const router = express.Router();

// POST /api/register - Register or update profile
router.post("/register", (req, res) => {
  const { description, avatar_url, appear_offline } = req.body || {};
  try {
    db.upsertUser(req.username, description, avatar_url);
    db.setOnline(req.username, !!appear_offline);
    return res.json({ result: "ok", username: req.username });
  } catch (e) {
    return res.status(500).json({ error: "Registration failed", details: e.message });
  }
});

// POST /api/heartbeat - Keep-alive ping
router.post("/heartbeat", (req, res) => {
  const { appear_offline } = req.body || {};
  try {
    db.setOnline(req.username, !!appear_offline);
    // Return unread count as a convenience
    const unread = db.getUnreadMessages(req.username);
    return res.json({ result: "ok", unread_count: unread.length });
  } catch (e) {
    return res.status(500).json({ error: "Heartbeat failed" });
  }
});

// GET /api/users/online - List online users
router.get("/users/online", (req, res) => {
  try {
    const users = db.getOnlineUsers();
    return res.json({ users });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch online users" });
  }
});

// GET /api/users/:username - Get a user's profile
router.get("/users/:username", (req, res) => {
  const targetUsername = req.params.username;
  if (!targetUsername || typeof targetUsername !== "string") {
    return res.status(400).json({ error: "Invalid username" });
  }
  try {
    const user = db.getUser(targetUsername);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    return res.json({
      username: user.username,
      description: user.description,
      avatar_url: user.avatar_url,
      is_online: !!user.is_online,
      created_at: user.created_at,
    });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch user" });
  }
});

module.exports = router;
