const express = require("express");
const db = require("../database");
const { adminMiddleware } = require("../auth");

const router = express.Router();

// All admin routes require admin key
router.use(adminMiddleware);

// GET /api/admin/users - List all registered users
router.get("/users", (req, res) => {
  try {
    const users = db.getDb().prepare("SELECT username, description, is_online, created_at, last_heartbeat FROM users ORDER BY username").all();
    return res.json({ users });
  } catch (e) {
    return res.status(500).json({ error: "Failed to list users" });
  }
});

// GET /api/admin/stats - Server statistics
router.get("/stats", (req, res) => {
  try {
    const d = db.getDb();
    const userCount = d.prepare("SELECT COUNT(*) as count FROM users").get().count;
    const onlineCount = d.prepare("SELECT COUNT(*) as count FROM users WHERE is_online = 1").get().count;
    const messageCount = d.prepare("SELECT COUNT(*) as count FROM messages").get().count;
    const friendCount = d.prepare("SELECT COUNT(*) as count FROM friends WHERE status = 'accepted'").get().count;
    return res.json({
      users: userCount,
      online: onlineCount,
      messages: messageCount,
      friendships: friendCount,
    });
  } catch (e) {
    return res.status(500).json({ error: "Failed to get stats" });
  }
});

// POST /api/admin/purge-messages - Manually purge old messages
router.post("/purge-messages", (req, res) => {
  const { days } = req.body || {};
  if (!days || typeof days !== "number" || days <= 0) {
    return res.status(400).json({ error: "Provide a valid 'days' number > 0" });
  }
  try {
    db.purgeOldMessages(days);
    return res.json({ result: "ok", message: `Purged messages older than ${days} days` });
  } catch (e) {
    return res.status(500).json({ error: "Failed to purge messages" });
  }
});

module.exports = router;
