const express = require("express");
const crypto = require("crypto");
const { promisify } = require("util");
const db = require("../database");
const { authenticateRequest, getSessionUser } = require("../auth");

const scryptAsync = promisify(crypto.scrypt);
const router = express.Router();        // Pre-auth routes (register, login)
const authedRouter = express.Router();  // Routes requiring authentication

const SESSION_TTL_HOURS = 168; // 7 days

async function hashPassword(password) {
  const salt = crypto.randomBytes(32).toString("hex");
  const derived = await scryptAsync(password, salt, 64);
  return `${salt}:${derived.toString("hex")}`;
}

async function verifyPassword(password, stored) {
  const [salt, hash] = stored.split(":");
  const derived = await scryptAsync(password, salt, 64);
  const derivedHex = derived.toString("hex");
  return crypto.timingSafeEqual(Buffer.from(hash), Buffer.from(derivedHex));
}

function generateToken() {
  return crypto.randomBytes(48).toString("base64url");
}

// POST /api/register - Register a new account with password, or update profile if already authenticated
router.post("/register", async (req, res) => {
  const { username, password, description, avatar_url } = req.body || {};

  // Check if this is an API-key authenticated request (keyed mode)
  const apiKeyUser = authenticateRequest(req);
  if (apiKeyUser) {
    try {
      db.upsertUser(apiKeyUser, description, avatar_url);
      db.setOnline(apiKeyUser, false);
      return res.json({ result: "ok", username: apiKeyUser });
    } catch (e) {
      return res.status(500).json({ error: "Registration failed" });
    }
  }

  // Check if this is a session-token authenticated request (profile update)
  const sessionUser = getSessionUser(req);
  if (sessionUser) {
    try {
      db.upsertUser(sessionUser, description, avatar_url);
      return res.json({ result: "ok", username: sessionUser });
    } catch (e) {
      return res.status(500).json({ error: "Profile update failed" });
    }
  }

  // Open mode: require username and password for new accounts
  if (!username || typeof username !== "string" || username.length < 1 || username.length > 32) {
    return res.status(400).json({
      error: "Invalid username",
      message: "Username must be 1-32 characters.",
    });
  }

  if (!password || typeof password !== "string" || password.length < 6) {
    return res.status(400).json({
      error: "Password required",
      message: "Password must be at least 6 characters.",
    });
  }

  try {
    const existingUser = db.getUser(username);
    if (existingUser && existingUser.password_hash) {
      return res.status(409).json({
        error: "Username taken",
        message: "This username is already registered. Use /api/login to sign in.",
      });
    }

    const hash = await hashPassword(password);
    db.upsertUser(username, description, avatar_url);
    db.setPasswordHash(username, hash);
    db.setOnline(username, false);

    // Return a session token so the client can immediately start using the API
    const token = generateToken();
    const expiresAt = new Date(Date.now() + SESSION_TTL_HOURS * 3600000).toISOString().replace("T", " ").split(".")[0];
    db.createSession(token, username, expiresAt);

    return res.json({ result: "ok", username, token });
  } catch (e) {
    return res.status(500).json({ error: "Registration failed" });
  }
});

// POST /api/login - Authenticate with password, returns session token
router.post("/login", async (req, res) => {
  const { username, password } = req.body || {};

  if (!username || !password) {
    return res.status(400).json({
      error: "Missing credentials",
      message: "Both username and password are required.",
    });
  }

  try {
    const user = db.getUser(username);
    if (!user || !user.password_hash) {
      return res.status(401).json({ error: "Invalid username or password" });
    }

    const valid = await verifyPassword(password, user.password_hash);
    if (!valid) {
      return res.status(401).json({ error: "Invalid username or password" });
    }

    const token = generateToken();
    const expiresAt = new Date(Date.now() + SESSION_TTL_HOURS * 3600000).toISOString().replace("T", " ").split(".")[0];
    db.createSession(token, username, expiresAt);

    return res.json({ result: "ok", username, token });
  } catch (e) {
    return res.status(500).json({ error: "Login failed" });
  }
});

// POST /api/heartbeat - Keep-alive ping
authedRouter.post("/heartbeat", (req, res) => {
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
authedRouter.get("/users/online", (req, res) => {
  try {
    const users = db.getOnlineUsers();
    return res.json({ users });
  } catch (e) {
    return res.status(500).json({ error: "Failed to fetch online users" });
  }
});

// GET /api/users/:username - Get a user's profile
authedRouter.get("/users/:username", (req, res) => {
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

module.exports = { preAuth: router, authed: authedRouter };
