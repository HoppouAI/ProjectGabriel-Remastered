const Database = require("better-sqlite3");
const path = require("path");
const fs = require("fs");

let db = null;

function initDatabase(dbPath) {
  const fullPath = path.resolve(__dirname, "..", dbPath);
  const dir = path.dirname(fullPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  db = new Database(fullPath);

  // Enable WAL mode for better concurrent read performance
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  // Create tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      description TEXT DEFAULT '',
      avatar_url TEXT DEFAULT '',
      created_at TEXT DEFAULT (datetime('now')),
      last_heartbeat TEXT DEFAULT NULL,
      is_online INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      from_user TEXT NOT NULL,
      to_user TEXT NOT NULL,
      content TEXT NOT NULL,
      timestamp TEXT DEFAULT (datetime('now')),
      read INTEGER DEFAULT 0,
      FOREIGN KEY (from_user) REFERENCES users(username),
      FOREIGN KEY (to_user) REFERENCES users(username)
    );

    CREATE TABLE IF NOT EXISTS friends (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user1 TEXT NOT NULL,
      user2 TEXT NOT NULL,
      status TEXT CHECK(status IN ('pending', 'accepted', 'denied')) DEFAULT 'pending',
      requested_by TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (user1) REFERENCES users(username),
      FOREIGN KEY (user2) REFERENCES users(username),
      UNIQUE(user1, user2)
    );

    CREATE TABLE IF NOT EXISTS blocks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      blocker TEXT NOT NULL,
      blocked TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (blocker) REFERENCES users(username),
      FOREIGN KEY (blocked) REFERENCES users(username),
      UNIQUE(blocker, blocked)
    );

    CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_user);
    CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_user);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
    CREATE INDEX IF NOT EXISTS idx_messages_read ON messages(to_user, read);
    CREATE INDEX IF NOT EXISTS idx_friends_users ON friends(user1, user2);
    CREATE INDEX IF NOT EXISTS idx_blocks_blocker ON blocks(blocker);
  `);

  return db;
}

function getDb() {
  if (!db) {
    throw new Error("Database not initialized. Call initDatabase() first.");
  }
  return db;
}

// ── User operations ──

function upsertUser(username, description, avatarUrl) {
  const stmt = getDb().prepare(`
    INSERT INTO users (username, description, avatar_url)
    VALUES (?, ?, ?)
    ON CONFLICT(username) DO UPDATE SET
      description = excluded.description,
      avatar_url = excluded.avatar_url
  `);
  return stmt.run(username, description || "", avatarUrl || "");
}

function getUser(username) {
  return getDb().prepare("SELECT * FROM users WHERE username = ?").get(username);
}

function setOnline(username, appearOffline) {
  if (appearOffline) {
    // Update heartbeat but stay offline in listings
    getDb().prepare(`
      UPDATE users SET is_online = 0, last_heartbeat = datetime('now') WHERE username = ?
    `).run(username);
  } else {
    getDb().prepare(`
      UPDATE users SET is_online = 1, last_heartbeat = datetime('now') WHERE username = ?
    `).run(username);
  }
}

function setOffline(username) {
  getDb().prepare("UPDATE users SET is_online = 0 WHERE username = ?").run(username);
}

function getOnlineUsers() {
  return getDb().prepare("SELECT username, description, avatar_url, last_heartbeat FROM users WHERE is_online = 1").all();
}

function cleanStalePresence(timeoutSeconds) {
  getDb().prepare(`
    UPDATE users SET is_online = 0
    WHERE is_online = 1 AND last_heartbeat IS NOT NULL
    AND (julianday('now') - julianday(last_heartbeat)) * 86400 > ?
  `).run(timeoutSeconds);
}

// ── Message operations ──

function sendMessage(fromUser, toUser, content) {
  const stmt = getDb().prepare(`
    INSERT INTO messages (from_user, to_user, content) VALUES (?, ?, ?)
  `);
  return stmt.run(fromUser, toUser, content);
}

function getRecentMessages(username, limit) {
  return getDb().prepare(`
    SELECT id, from_user, to_user, content, timestamp, read
    FROM messages
    WHERE to_user = ? OR from_user = ?
    ORDER BY timestamp DESC
    LIMIT ?
  `).all(username, username, limit || 50);
}

function getMessagesByUser(username, otherUser, limit) {
  return getDb().prepare(`
    SELECT id, from_user, to_user, content, timestamp, read
    FROM messages
    WHERE (from_user = ? AND to_user = ?) OR (from_user = ? AND to_user = ?)
    ORDER BY timestamp DESC
    LIMIT ?
  `).all(username, otherUser, otherUser, username, limit || 50);
}

function getUnreadMessages(username) {
  return getDb().prepare(`
    SELECT id, from_user, to_user, content, timestamp, read
    FROM messages
    WHERE to_user = ? AND read = 0
    ORDER BY timestamp DESC
  `).all(username);
}

function markMessagesRead(username, fromUser) {
  if (fromUser) {
    return getDb().prepare(`
      UPDATE messages SET read = 1 WHERE to_user = ? AND from_user = ? AND read = 0
    `).run(username, fromUser);
  }
  return getDb().prepare(`
    UPDATE messages SET read = 1 WHERE to_user = ? AND read = 0
  `).run(username);
}

function purgeOldMessages(days) {
  if (days <= 0) return;
  getDb().prepare(`
    DELETE FROM messages WHERE (julianday('now') - julianday(timestamp)) > ?
  `).run(days);
}

// ── Friend operations ──

function sendFriendRequest(fromUser, toUser) {
  // Normalize order: user1 is alphabetically first
  const [user1, user2] = [fromUser, toUser].sort();
  const existing = getDb().prepare(
    "SELECT * FROM friends WHERE user1 = ? AND user2 = ?"
  ).get(user1, user2);

  if (existing) {
    if (existing.status === "accepted") {
      return { error: "Already friends" };
    }
    if (existing.status === "pending" && existing.requested_by === fromUser) {
      return { error: "Friend request already sent" };
    }
    if (existing.status === "pending" && existing.requested_by !== fromUser) {
      // Other user already sent a request, auto-accept
      getDb().prepare(`
        UPDATE friends SET status = 'accepted', updated_at = datetime('now')
        WHERE user1 = ? AND user2 = ?
      `).run(user1, user2);
      return { result: "accepted", message: "Mutual request - automatically accepted" };
    }
    if (existing.status === "denied") {
      // Allow re-sending after deny
      getDb().prepare(`
        UPDATE friends SET status = 'pending', requested_by = ?, updated_at = datetime('now')
        WHERE user1 = ? AND user2 = ?
      `).run(fromUser, user1, user2);
      return { result: "sent" };
    }
  }

  getDb().prepare(`
    INSERT INTO friends (user1, user2, requested_by) VALUES (?, ?, ?)
  `).run(user1, user2, fromUser);
  return { result: "sent" };
}

function acceptFriendRequest(username, fromUser) {
  const [user1, user2] = [username, fromUser].sort();
  const existing = getDb().prepare(
    "SELECT * FROM friends WHERE user1 = ? AND user2 = ? AND status = 'pending' AND requested_by = ?"
  ).get(user1, user2, fromUser);

  if (!existing) {
    return { error: "No pending request from that user" };
  }

  getDb().prepare(`
    UPDATE friends SET status = 'accepted', updated_at = datetime('now')
    WHERE user1 = ? AND user2 = ?
  `).run(user1, user2);
  return { result: "accepted" };
}

function denyFriendRequest(username, fromUser) {
  const [user1, user2] = [username, fromUser].sort();
  const existing = getDb().prepare(
    "SELECT * FROM friends WHERE user1 = ? AND user2 = ? AND status = 'pending' AND requested_by = ?"
  ).get(user1, user2, fromUser);

  if (!existing) {
    return { error: "No pending request from that user" };
  }

  getDb().prepare(`
    UPDATE friends SET status = 'denied', updated_at = datetime('now')
    WHERE user1 = ? AND user2 = ?
  `).run(user1, user2);
  return { result: "denied" };
}

function removeFriend(username, otherUser) {
  const [user1, user2] = [username, otherUser].sort();
  const existing = getDb().prepare(
    "SELECT * FROM friends WHERE user1 = ? AND user2 = ? AND status = 'accepted'"
  ).get(user1, user2);

  if (!existing) {
    return { error: "Not friends with that user" };
  }

  getDb().prepare("DELETE FROM friends WHERE user1 = ? AND user2 = ?").run(user1, user2);
  return { result: "removed" };
}

function listFriends(username) {
  return getDb().prepare(`
    SELECT
      CASE WHEN user1 = ? THEN user2 ELSE user1 END AS friend,
      status, created_at, updated_at
    FROM friends
    WHERE (user1 = ? OR user2 = ?) AND status = 'accepted'
  `).all(username, username, username);
}

function getPendingRequests(username) {
  return getDb().prepare(`
    SELECT
      requested_by AS from_user,
      CASE WHEN user1 = requested_by THEN user2 ELSE user1 END AS to_user,
      created_at
    FROM friends
    WHERE (user1 = ? OR user2 = ?) AND status = 'pending' AND requested_by != ?
  `).all(username, username, username);
}

function getSentRequests(username) {
  return getDb().prepare(`
    SELECT
      CASE WHEN user1 = ? THEN user2 ELSE user1 END AS to_user,
      created_at
    FROM friends
    WHERE (user1 = ? OR user2 = ?) AND status = 'pending' AND requested_by = ?
  `).all(username, username, username, username);
}

// ── Block operations ──

function blockUser(blocker, blocked) {
  try {
    getDb().prepare("INSERT INTO blocks (blocker, blocked) VALUES (?, ?)").run(blocker, blocked);
  } catch (e) {
    if (e.message.includes("UNIQUE constraint")) {
      return { error: "Already blocked" };
    }
    throw e;
  }
  // Also remove any friendship
  const [user1, user2] = [blocker, blocked].sort();
  getDb().prepare("DELETE FROM friends WHERE user1 = ? AND user2 = ?").run(user1, user2);
  return { result: "blocked" };
}

function unblockUser(blocker, blocked) {
  const result = getDb().prepare("DELETE FROM blocks WHERE blocker = ? AND blocked = ?").run(blocker, blocked);
  if (result.changes === 0) {
    return { error: "User was not blocked" };
  }
  return { result: "unblocked" };
}

function isBlocked(user1, user2) {
  const row = getDb().prepare(
    "SELECT 1 FROM blocks WHERE (blocker = ? AND blocked = ?) OR (blocker = ? AND blocked = ?)"
  ).get(user1, user2, user2, user1);
  return !!row;
}

function getBlockedUsers(username) {
  return getDb().prepare(
    "SELECT blocked, created_at FROM blocks WHERE blocker = ?"
  ).all(username);
}

module.exports = {
  initDatabase,
  getDb,
  upsertUser,
  getUser,
  setOnline,
  setOffline,
  getOnlineUsers,
  cleanStalePresence,
  sendMessage,
  getRecentMessages,
  getMessagesByUser,
  getUnreadMessages,
  markMessagesRead,
  purgeOldMessages,
  sendFriendRequest,
  acceptFriendRequest,
  denyFriendRequest,
  removeFriend,
  listFriends,
  getPendingRequests,
  getSentRequests,
  blockUser,
  unblockUser,
  isBlocked,
  getBlockedUsers,
};
