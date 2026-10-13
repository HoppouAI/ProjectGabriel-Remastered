"""Built in call type for each tool on models where blocking vs background is optional."""

# fire and forget, the model already said what it is doing so the result stays quiet
_ACTIONS = {
    "emotion", "stopAnimation", "playMotion", "playMotionSequence", "stopMotion", "resetPose",
    "vrchatMove", "vrchatStop", "vrchatJump", "vrchatCrouch", "vrchatCrawl", "vrchatDrop",
    "vrchatLook", "vrchatLookVertical", "vrchatGrab", "vrchatUse", "boopPlayer",
    "vrchatSetAvatarScale", "vrchatAdjustAvatarScale", "setMoveSpeed",
    "startFollow", "stopFollow", "setFollowDistance", "startWander", "stopWander",
    "gotoWaypoint", "saveWaypoint", "deleteWaypoint", "cancelWalk",
    "playMusic", "pauseMusic", "resumeMusic", "stopMusic", "setMusicVolume",
    "startMusicGen", "steerMusicGen", "pauseMusicGen", "resumeMusicGen", "stopMusicGen", "setMusicGenVolume",
    "playSoundboard", "playRandomSoundboard", "stopSoundboard",
    "setVoiceBoost", "setVoicePitch", "toggleLowQualityMic", "toggleVrchatMic", "enableYapMode",
    "switchAvatar", "switchPersonality", "switchTTSProvider", "updateStatus",
    "saveMemory", "updateMemory", "deleteMemory",
    "relayToDiscord", "sendDiscordMessage", "discord_hangUp",
    "invitePlayer", "requestInvite", "inviteSelfToInstance",
    "sendFriendRequest", "acceptFriendRequest", "declineFriendRequest",
    "socialSendMessage", "socialSendFriendRequest", "socialAcceptFriend", "socialDenyFriend",
    "socialRemoveFriend", "socialBlockUser", "socialUnblockUser",
}

# local and quick, the model needs the answer before it can say anything useful
_QUICK_LOOKUPS = {
    "getCurrentTime", "getTimeInZone", "getCurrentStatus", "getCurrentPersonality",
    "listPersonalities", "listMusic", "listVoices", "listTTSProviders", "listMemories",
    "searchMemories", "searchSoundboard", "getSystemSpecs", "vrchatGetAvatarScale",
    "getDiscordStatus", "getYapModeStatus", "listWaypoints", "searchTools",
}

# network or model backed, run it in the background and bring it up when done
_SLOW_LOOKUPS = {
    "webSearch", "readWebpage", "recallMemories",
    "getFriendInfo", "getMutualFriends", "getInstancePlayers", "getAvatarInfo", "getOwnAvatar",
    "searchAvatars", "searchWorlds", "listFriendRequests",
    "socialFetchMessages", "socialFetchMessagesByUser", "socialGetOnline",
    "socialGetPendingRequests", "socialGetProfile", "socialListFriends",
}


def default_call_type(name: str):
    """(type, scheduling) for a tool, or (None, None) to leave it to the model."""
    if name in _ACTIONS:
        return "non_blocking", "silent"
    if name in _QUICK_LOOKUPS:
        return "blocking", None
    if name in _SLOW_LOOKUPS:
        return "non_blocking", "when_idle"
    return None, None
