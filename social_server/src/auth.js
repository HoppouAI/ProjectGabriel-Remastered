const crypto = require("crypto");
const { logAuth } = require("./logger");

// Build a key->username lookup from config
let keyMap = new Map();
let adminKey = null;
let openMode = false;
let requiredUaPrefix = "";
let _db = null;

function initAuth(config) {
  keyMap = new Map();
  openMode = config.openMode || false;
  requiredUaPrefix = config.requiredUserAgentPrefix || "";
  for (const entry of config.apiKeys) {
    keyMap.set(entry.key, entry.username);
  }
  adminKey = config.adminKey;
}

// Set the database module reference (called after both auth and db are initialized)
function setAuthDb(dbModule) {
  _db = dbModule;
}

function _getClientIp(req) {
  return req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.socket?.remoteAddress || "unknown";
}

function _getUserAgent(req) {
  return req.headers["user-agent"] || "";
}

function _extractBearerToken(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader) return null;
  const parts = authHeader.split(" ");
  if (parts.length !== 2 || parts[0] !== "Bearer") return null;
  return parts[1];
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

  // Try API key auth first (works in both modes)
  const apiKeyUsername = authenticateRequest(req);
  if (apiKeyUsername) {
    req.username = apiKeyUsername;
    req.apiKeyAuth = true;
    logAuth("SUCCESS", { username: apiKeyUsername, ip, userAgent: ua });
    return next();
  }

  // Try session token auth (works in both modes for password-based users)
  const token = _extractBearerToken(req);
  if (token && _db) {
    const session = _db.getSession(token);
    if (session) {
      req.username = session.username;
      req.apiKeyAuth = false;
      return next();
    }
  }

  if (openMode) {
    logAuth("REJECTED_OPEN", { ip, userAgent: ua, reason: "No valid session token" });
    return res.status(401).json({
      error: "Authentication required",
      message: "Register at /api/register or login at /api/login to get a session token. " +
               "Then include it as: Authorization: Bearer <token>",
    });
  }

  // Keyed mode: neither API key nor session token worked
  logAuth("REJECTED_KEY", { ip, userAgent: ua, reason: "Invalid or missing API key" });
  console.log(`[Auth] Rejected ${req.method} ${req.path} from ${ip} - invalid API key (UA: "${ua}")`);
  return res.status(401).json({
    error: "Authentication failed",
    message: "Invalid or missing API key or session token. Include your API key in the Authorization header as: " +
             "Authorization: Bearer <your-api-key>. Or login at /api/login to get a session token.",
  });
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

// Check for a valid session token and return the username, or null
function getSessionUser(req) {
  const token = _extractBearerToken(req);
  if (!token || !_db) return null;
  const session = _db.getSession(token);
  return session ? session.username : null;
}

module.exports = { initAuth, setAuthDb, authenticateRequest, getSessionUser, isAdmin, userAgentMiddleware, authMiddleware, adminMiddleware };
