const http = require("http");
const express = require("express");
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const { loadConfig } = require("./src/config");
const { initDatabase, cleanStalePresence, purgeOldMessages } = require("./src/database");
const { initAuth, userAgentMiddleware, authMiddleware, authenticateRequest } = require("./src/auth");
const { initLogger, logInfo, closeLogger } = require("./src/logger");
const { WebSocketManager } = require("./src/websocket");
const usersRouter = require("./src/routes/users");
const messagesRouter = require("./src/routes/messages");
const friendsRouter = require("./src/routes/friends");
const adminRouter = require("./src/routes/admin");

// ── Load config ──
const config = loadConfig();
console.log(`[Social Server] Loading config...`);

// ── Initialize logger ──
initLogger(config.logging);
logInfo("Server starting up");

// ── Initialize database ──
initDatabase(config.dbPath);
console.log(`[Social Server] Database initialized at ${config.dbPath}`);
logInfo(`Database initialized at ${config.dbPath}`);

// ── Initialize auth ──
initAuth(config);

// ── Express app ──
const app = express();

// Security headers
app.use(helmet());

// ── Base path router ──
// When running behind a reverse proxy with a path prefix (e.g. /social/),
// set server.base_path in config.yml so routes mount correctly.
const basePath = config.basePath;
const router = express.Router();

// Body parsing with size limit (inside router so errors hit our handler)
router.use(express.json({ limit: "16kb" }));

// User-Agent validation (applies to all routes including /health)
router.use(userAgentMiddleware);

// Rate limiting (per API key, falling back to IP)
const limiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMax,
  keyGenerator: (req) => {
    // Rate limit by API key (username) if authenticated, otherwise by IP
    const username = authenticateRequest(req);
    return username || req.ip;
  },
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Rate limit exceeded",
    message: "Too many requests. Please slow down and try again in a moment.",
  },
});
router.use("/api/", limiter);

// Health check (no auth)
router.get("/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime() });
});

// All /api routes require authentication (except admin which has its own)
router.use("/api/admin", adminRouter);
router.use("/api", authMiddleware);
router.use("/api", usersRouter);
router.use("/api", messagesRouter);
router.use("/api", friendsRouter);

// 404 handler
router.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

// Error handler (never expose stack traces)
router.use((err, req, res, _next) => {
  if (err.type === "entity.parse.failed") {
    res.status(400).json({ error: "Invalid JSON", message: "Request body contains malformed JSON." });
    return;
  }
  console.error("[Social Server] Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

// Mount router at base path
app.use(basePath, router);

// ── Create HTTP server ──
const server = http.createServer(app);

// ── WebSocket ──
const wsManager = new WebSocketManager();
wsManager.attach(server, config);
app.set("wsManager", wsManager);

// ── Set message length from config ──
messagesRouter.setMaxMessageLength(config.maxMessageLength);

// ── Periodic tasks ──
// Clean stale presence every 30 seconds
setInterval(() => {
  cleanStalePresence(config.heartbeatTimeout);
}, 30000);

// Purge old messages based on retention policy
if (config.retentionDays > 0) {
  // Run daily
  setInterval(() => {
    purgeOldMessages(config.retentionDays);
    console.log(`[Social Server] Purged messages older than ${config.retentionDays} days`);
  }, 86400000); // 24 hours
  // Also run once at startup
  purgeOldMessages(config.retentionDays);
}

// ── Start server ──
server.listen(config.port, config.host, () => {
  const W = 50; // inner width between box edges
  const pad = (s) => s + " ".repeat(Math.max(0, W - s.length));
  const line = (s) => `  ║${pad(s)}║`;
  const bar = (l, r, fill = "═") => `  ${l}${fill.repeat(W)}${r}`;

  const mode = config.openMode ? "OPEN (no key required)" : `${config.apiKeys.length} API key(s)`;

  const bp = basePath === "/" ? "" : basePath;
  console.log("");
  console.log(bar("╔", "╗"));
  console.log(line("   ProjectGabriel Social Server v1.0.0"));
  console.log(bar("╠", "╣"));
  console.log(line(`  HTTP:   http://${config.host}:${config.port}${bp}`));
  console.log(line(`  WS:     ws://${config.host}:${config.port}${bp}/ws`));
  console.log(line(`  Auth:   ${mode}`));
  if (bp) console.log(line(`  Base:   ${bp}`));
  console.log(line(`  Log:    ${config.logging.enabled ? config.logging.path : "disabled"}`));
  console.log(bar("╚", "╝"));
  console.log("");

  logInfo(`Server listening on ${config.host}:${config.port} (mode: ${config.openMode ? "open" : "key-auth"})`);
});

// ── Graceful shutdown ──
function shutdown() {
  console.log("\n[Social Server] Shutting down...");
  logInfo("Server shutting down");
  wsManager.shutdown();
  server.close(() => {
    console.log("[Social Server] Server closed");
    closeLogger();
    process.exit(0);
  });
  // Force exit after 5s
  setTimeout(() => {
    closeLogger();
    process.exit(1);
  }, 5000);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
