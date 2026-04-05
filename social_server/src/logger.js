const fs = require("fs");
const path = require("path");

let logStream = null;
let logLevel = "info";

const LEVELS = { error: 0, warn: 1, info: 2 };

function initLogger(config) {
  logLevel = config.level || "info";
  if (!config.enabled) return;

  const logPath = path.resolve(__dirname, "..", config.path);
  const logDir = path.dirname(logPath);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  logStream = fs.createWriteStream(logPath, { flags: "a" });
}

function _timestamp() {
  return new Date().toISOString();
}

function _write(level, message) {
  if (LEVELS[level] > LEVELS[logLevel]) return;
  const line = `[${_timestamp()}] [${level.toUpperCase()}] ${message}`;
  if (logStream) {
    logStream.write(line + "\n");
  }
}

function logAuth(event, details) {
  const parts = [`AUTH ${event}`];
  if (details.username) parts.push(`user=${details.username}`);
  if (details.ip) parts.push(`ip=${details.ip}`);
  if (details.userAgent) parts.push(`ua="${details.userAgent}"`);
  if (details.reason) parts.push(`reason="${details.reason}"`);
  _write("info", parts.join(" | "));
}

function logWarn(message) {
  _write("warn", message);
}

function logError(message) {
  _write("error", message);
}

function logInfo(message) {
  _write("info", message);
}

function closeLogger() {
  if (logStream) {
    logStream.end();
    logStream = null;
  }
}

module.exports = { initLogger, logAuth, logWarn, logError, logInfo, closeLogger };
