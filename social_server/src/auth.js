const crypto = require("crypto");
const { logAuth } = require("./logger");

// Build a key->username lookup from config
let keyMap = new Map();
let adminKey = null;
let openMode = false;
let requiredUaPrefix = "";

function initAuth(config) {
  keyMap = new Map();
  openMode = config.openMode || false;
  requiredUaPrefix = config.requiredUserAgentPrefix || "";
  for (const entry of config.apiKeys) {
    keyMap.set(entry.key, entry.username);
  }
  adminKey = config.adminKey;
}

function _getClientIp(req) {
  return req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.socket?.remoteAddress || "unknown";
}

function _getUserAgent(req) {
  return req.headers["user-agent"] || "";
}

function authenticateRequest(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader) return null;

  // Expect "Bearer <key>"
  const parts = authHeader.split(" ");
  if (parts.length !== 2 || parts[0] !== "Bearer") return null;

  const key = parts[1];
  const username = keyMap.get(key);
  return username || null;
}

function isAdmin(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader) return false;
  const parts = authHeader.split(" ");
  if (parts.length !== 2 || parts[0] !== "Bearer") return false;
  // Use timing-safe comparison to prevent timing attacks
  const provided = Buffer.from(parts[1]);
  const expected = Buffer.from(adminKey);
  if (provided.length !== expected.length) return false;
  return crypto.timingSafeEqual(provided, expected);
}

// Middleware: validate User-Agent header
function userAgentMiddleware(req, res, next) {
  if (!requiredUaPrefix) return next();

  const ua = _getUserAgent(req);
  if (!ua || !ua.startsWith(requiredUaPrefix)) {
    const ip = _getClientIp(req);
    logAuth("REJECTED_UA", { ip, userAgent: ua || "(none)", reason: "Invalid User-Agent" });
    console.log(`[Auth] Rejected connection from ${ip} - invalid User-Agent: "${ua || "(none)"}""`);
    return res.status(403).json({
      error: "Connection refused",
      message: `Invalid User-Agent. All clients must identify with a User-Agent header starting with "${requiredUaPrefix}". ` +
               `Expected format: ${requiredUaPrefix}<name>/<version> (e.g. ${requiredUaPrefix}MyBot/1.0)`,
    });
  }
  next();
}

// Express middleware - authenticates and sets req.username
function authMiddleware(req, res, next) {
  const ip = _getClientIp(req);
  const ua = _getUserAgent(req);

  if (openMode) {
    // In open mode, username comes from body (for POST) or query (for GET)
    const username = req.body?.username || req.query?.username;
    if (username && typeof username === "string" && username.length >= 1 && username.length <= 32) {
      req.username = username;
      return next();
    }
    // Also allow authenticated requests in open mode (for users who still configure keys)
    const authUsername = authenticateRequest(req);
    if (authUsername) {
      req.username = authUsername;
      logAuth("SUCCESS", { username: authUsername, ip, userAgent: ua });
      return next();
    }
    logAuth("REJECTED_OPEN", { ip, userAgent: ua, reason: "No username provided in open mode" });
    return res.status(400).json({
      error: "Username required",
      message: "In open mode, include a \"username\" field in your request body (POST) or query parameter (GET). " +
               "Username must be 1-32 characters.",
    });
  }

  // Standard API key auth
  const username = authenticateRequest(req);
  if (!username) {
    logAuth("REJECTED_KEY", { ip, userAgent: ua, reason: "Invalid or missing API key" });
    console.log(`[Auth] Rejected ${req.method} ${req.path} from ${ip} - invalid API key (UA: "${ua}")`);
    return res.status(401).json({
      error: "Authentication failed",
      message: "Invalid or missing API key. Include your API key in the Authorization header as: " +
               "Authorization: Bearer <your-api-key>. Contact the server administrator if you need a key.",
    });
  }
  req.username = username;
  logAuth("SUCCESS", { username, ip, userAgent: ua });
  next();
}

// Express middleware - requires admin key
function adminMiddleware(req, res, next) {
  const ip = _getClientIp(req);
  const ua = _getUserAgent(req);
  if (!isAdmin(req)) {
    logAuth("REJECTED_ADMIN", { ip, userAgent: ua, reason: "Invalid admin key" });
    return res.status(403).json({
      error: "Admin access required",
      message: "This endpoint requires the admin key. Include it in the Authorization header as: " +
               "Authorization: Bearer <admin-key>.",
    });
  }
  logAuth("ADMIN_ACCESS", { ip, userAgent: ua });
  next();
}

module.exports = { initAuth, authenticateRequest, isAdmin, userAgentMiddleware, authMiddleware, adminMiddleware };
