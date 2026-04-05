const express = require("express");
const db = require("../database");

const router = express.Router();

// POST /api/friends/request - Send a friend request
router.post("/friends/request", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }
  if (username === req.username) {
    return res.status(400).json({ error: "Cannot send a friend request to yourself" });
  }

  const targetUser = db.getUser(username);
  if (!targetUser) {
    return res.status(404).json({ error: `User "${username}" not found` });
  }

  if (db.isBlocked(req.username, username)) {
    return res.status(403).json({ error: "Cannot send friend request - blocked" });
  }

  try {
    const result = db.sendFriendRequest(req.username, username);
    if (result.error) {
      return res.status(409).json({ error: result.error });
    }

    // Notify via WebSocket
    const wsManager = req.app.get("wsManager");
    if (wsManager) {
      wsManager.notifyUser(username, {
        type: "friend_request",
        from: req.username,
      });
      // If auto-accepted (mutual request), notify both
      if (result.result === "accepted") {
        wsManager.notifyUser(req.username, {
          type: "friend_accepted",
          username: username,
        });
        wsManager.notifyUser(username, {
          type: "friend_accepted",
          username: req.username,
        });
      }
    }

    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to send friend request", details: e.message });
  }
});

// POST /api/friends/accept - Accept a friend request
router.post("/friends/accept", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }

  try {
    const result = db.acceptFriendRequest(req.username, username);
    if (result.error) {
      return res.status(404).json({ error: result.error });
    }

    // Notify the requester
    const wsManager = req.app.get("wsManager");
    if (wsManager) {
      wsManager.notifyUser(username, {
        type: "friend_accepted",
        username: req.username,
      });
    }

    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to accept request", details: e.message });
  }
});

// POST /api/friends/deny - Deny a friend request
router.post("/friends/deny", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }

  try {
    const result = db.denyFriendRequest(req.username, username);
    if (result.error) {
      return res.status(404).json({ error: result.error });
    }
    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to deny request", details: e.message });
  }
});

// POST /api/friends/remove - Remove a friend
router.post("/friends/remove", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }

  try {
    const result = db.removeFriend(req.username, username);
    if (result.error) {
      return res.status(404).json({ error: result.error });
    }

    // Notify the other user
    const wsManager = req.app.get("wsManager");
    if (wsManager) {
      wsManager.notifyUser(username, {
        type: "friend_removed",
        username: req.username,
      });
    }

    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to remove friend", details: e.message });
  }
});

// GET /api/friends/list - List accepted friends
router.get("/friends/list", (req, res) => {
  try {
    const friends = db.listFriends(req.username);
    // Enrich with online status
    const enriched = friends.map((f) => {
      const user = db.getUser(f.friend);
      return {
        username: f.friend,
        description: user?.description || "",
        is_online: !!user?.is_online,
        friends_since: f.updated_at,
      };
    });
    return res.json({ friends: enriched });
  } catch (e) {
    return res.status(500).json({ error: "Failed to list friends" });
  }
});

// GET /api/friends/pending - Get pending friend requests (incoming)
router.get("/friends/pending", (req, res) => {
  try {
    const pending = db.getPendingRequests(req.username);
    return res.json({ requests: pending });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch pending requests" });
  }
});

// GET /api/friends/sent - Get sent friend requests (outgoing)
router.get("/friends/sent", (req, res) => {
  try {
    const sent = db.getSentRequests(req.username);
    return res.json({ requests: sent });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch sent requests" });
  }
});

// POST /api/friends/block - Block a user
router.post("/friends/block", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }
  if (username === req.username) {
    return res.status(400).json({ error: "Cannot block yourself" });
  }

  try {
    const result = db.blockUser(req.username, username);
    if (result.error) {
      return res.status(409).json({ error: result.error });
    }
    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to block user", details: e.message });
  }
});

// POST /api/friends/unblock - Unblock a user
router.post("/friends/unblock", (req, res) => {
  const { username } = req.body || {};
  if (!username || typeof username !== "string") {
    return res.status(400).json({ error: "Missing or invalid 'username' field" });
  }

  try {
    const result = db.unblockUser(req.username, username);
    if (result.error) {
      return res.status(404).json({ error: result.error });
    }
    return res.json(result);
  } catch (e) {
    return res.status(500).json({ error: "Failed to unblock user", details: e.message });
  }
});

// GET /api/friends/blocked - List blocked users
router.get("/friends/blocked", (req, res) => {
  try {
    const blocked = db.getBlockedUsers(req.username);
    return res.json({ blocked });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch blocked users" });
  }
});

module.exports = router;
