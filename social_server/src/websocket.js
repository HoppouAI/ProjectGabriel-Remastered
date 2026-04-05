const { WebSocketServer } = require("ws");
const { authenticateRequest } = require("./auth");
const { logAuth, logInfo } = require("./logger");

class WebSocketManager {
  constructor() {
    // Map of username -> Set of WebSocket connections
    this.clients = new Map();
    this.wss = null;
  }

  attach(server, config) {
    const basePath = (config.basePath || "").replace(/\/$/, "");
    this.wss = new WebSocketServer({ server, path: `${basePath}/ws` });
    const requiredUaPrefix = config.requiredUserAgentPrefix || "";

    this.wss.on("connection", (ws, req) => {
      const ip = req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.socket?.remoteAddress || "unknown";
      const ua = req.headers["user-agent"] || "";

      // Validate User-Agent
      if (requiredUaPrefix && (!ua || !ua.startsWith(requiredUaPrefix))) {
        logAuth("WS_REJECTED_UA", { ip, userAgent: ua || "(none)", reason: "Invalid User-Agent" });
        ws.close(4003, JSON.stringify({
          error: "Connection refused",
          message: `Invalid User-Agent. Expected format: ${requiredUaPrefix}<name>/<version>`,
        }));
        return;
      }

      // Authenticate the WebSocket connection via query param or header
      const username = this._authenticateWs(req, config);
      if (!username) {
        logAuth("WS_REJECTED_KEY", { ip, userAgent: ua, reason: "Invalid API key" });
        ws.close(4001, JSON.stringify({
          error: "Authentication failed",
          message: "Invalid or missing API key. Pass key as query param (?key=...) or Authorization header.",
        }));
        return;
      }

      logAuth("WS_CONNECTED", { username, ip, userAgent: ua });

      // Track connection
      if (!this.clients.has(username)) {
        this.clients.set(username, new Set());
      }
      this.clients.get(username).add(ws);
      console.log(`[WS] ${username} connected (${this.clients.get(username).size} connections)`);

      // Send welcome
      ws.send(JSON.stringify({ type: "connected", username }));

      // Heartbeat to detect stale connections
      ws.isAlive = true;
      ws.on("pong", () => { ws.isAlive = true; });

      ws.on("close", () => {
        const set = this.clients.get(username);
        if (set) {
          set.delete(ws);
          if (set.size === 0) {
            this.clients.delete(username);
          }
        }
        logInfo(`WS disconnected: ${username} from ${ip}`);
        console.log(`[WS] ${username} disconnected`);
      });

      ws.on("error", (err) => {
        console.error(`[WS] Error for ${username}:`, err.message);
      });
    });

    // Ping all connections every 30s to detect stale ones
    this._pingInterval = setInterval(() => {
      if (!this.wss) return;
      this.wss.clients.forEach((ws) => {
        if (ws.isAlive === false) {
          return ws.terminate();
        }
        ws.isAlive = false;
        ws.ping();
      });
    }, 30000);
  }

  _authenticateWs(req, config) {
    const url = new URL(req.url, "http://localhost");

    // In open mode, accept username from query param
    if (config.openMode) {
      const username = url.searchParams.get("username");
      if (username && username.length >= 1 && username.length <= 32) {
        return username;
      }
    }

    // Try query parameter: ?key=<api_key>
    const queryKey = url.searchParams.get("key");
    if (queryKey) {
      for (const entry of config.apiKeys) {
        if (entry.key === queryKey) {
          return entry.username;
        }
      }
    }

    // Try authorization header
    const fakeReq = { headers: req.headers };
    const { authenticateRequest: authReq } = require("./auth");
    return authReq(fakeReq);
  }

  notifyUser(username, payload) {
    const connections = this.clients.get(username);
    if (!connections || connections.size === 0) return false;

    const data = JSON.stringify(payload);
    for (const ws of connections) {
      if (ws.readyState === 1) { // OPEN
        ws.send(data);
      }
    }
    return true;
  }

  broadcast(payload, excludeUsername) {
    const data = JSON.stringify(payload);
    for (const [username, connections] of this.clients) {
      if (username === excludeUsername) continue;
      for (const ws of connections) {
        if (ws.readyState === 1) {
          ws.send(data);
        }
      }
    }
  }

  getConnectedUsernames() {
    return Array.from(this.clients.keys());
  }

  shutdown() {
    if (this._pingInterval) {
      clearInterval(this._pingInterval);
    }
    if (this.wss) {
      this.wss.close();
    }
  }
}

module.exports = { WebSocketManager };
