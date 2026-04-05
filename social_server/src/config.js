const fs = require("fs");
const path = require("path");
const yaml = require("js-yaml");

const CONFIG_PATH = path.join(__dirname, "..", "config.yml");

function loadConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    console.error("config.yml not found. Copy config.yml.example to config.yml and configure it.");
    process.exit(1);
  }
  const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
  const config = yaml.load(raw);

  // Validate required fields
  if (!config.security || !config.security.admin_key || config.security.admin_key === "CHANGE_ME_TO_A_SECURE_ADMIN_KEY") {
    console.error("ERROR: Please set a secure admin_key in config.yml");
    process.exit(1);
  }

  const openMode = !!config.security?.open_mode;

  // Only validate API keys when not in open mode
  if (!openMode) {
    if (!config.api_keys || config.api_keys.length === 0) {
      console.error("ERROR: No api_keys defined in config.yml (required when open_mode is false)");
      process.exit(1);
    }

    // Validate no duplicate usernames or keys
    const usernames = new Set();
    const keys = new Set();
    for (const entry of config.api_keys) {
      if (!entry.key || !entry.username) {
        console.error("ERROR: Each api_keys entry must have a key and username");
        process.exit(1);
      }
      if (entry.key.startsWith("CHANGE_ME")) {
        console.error(`ERROR: Please set a real API key for user "${entry.username}"`);
        process.exit(1);
      }
      if (usernames.has(entry.username)) {
        console.error(`ERROR: Duplicate username "${entry.username}" in api_keys`);
        process.exit(1);
      }
      if (keys.has(entry.key)) {
        console.error(`ERROR: Duplicate API key in api_keys`);
        process.exit(1);
      }
      usernames.add(entry.username);
      keys.add(entry.key);
    }
  }

  return {
    port: config.server?.port || 3000,
    host: config.server?.host || "0.0.0.0",
    adminKey: config.security.admin_key,
    openMode,
    rateLimitWindowMs: config.security?.rate_limit?.window_ms || 60000,
    rateLimitMax: config.security?.rate_limit?.max_requests || 100,
    requiredUserAgentPrefix: config.security?.required_user_agent_prefix ?? "ProjectGabrielSocial/",
    apiKeys: config.api_keys || [],
    dbPath: config.database?.path || "./data/social.sqlite",
    heartbeatTimeout: config.presence?.heartbeat_timeout || 60,
    maxMessageLength: config.messages?.max_length || 2000,
    retentionDays: config.messages?.retention_days || 30,
    logging: {
      enabled: config.logging?.enabled ?? true,
      path: config.logging?.path || "./data/server.log",
      level: config.logging?.level || "info",
    },
  };
}

module.exports = { loadConfig };
