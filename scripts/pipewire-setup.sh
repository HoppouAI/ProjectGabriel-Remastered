#!/usr/bin/env bash
# Virtual audio routing for ProjectGabriel on PipeWire, the linux stand-in for
# VB-Audio Virtual Cable. creates two null sinks so the AI and VRChat can hear
# each other:
#
#   Gabriel_Mic   the AI plays its TTS here. point VRChat's MIC at its monitor.
#   Gabriel_Ears  VRChat (and anything else) plays here. the AI listens on it.
#
# usage:
#   scripts/pipewire-setup.sh up      create the sinks (default)
#   scripts/pipewire-setup.sh down    remove them again
#   scripts/pipewire-setup.sh status  show what's loaded
set -u

STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}"
STATE_FILE="$STATE_DIR/gabriel-pipewire.modules"

need_pactl() {
    if ! command -v pactl >/dev/null 2>&1; then
        echo "ERROR: pactl not found. Install pipewire-pulse (or pulseaudio-utils)."
        exit 1
    fi
    if ! pactl info 2>/dev/null | grep -qi "PipeWire"; then
        echo "WARNING: PipeWire server not detected. This still works on plain"
        echo "         PulseAudio, but the rest of the guide assumes PipeWire."
    fi
}

cmd_up() {
    need_pactl
    if [ -f "$STATE_FILE" ]; then
        echo "Routing already set up. Run '$0 down' first to recreate it."
        exit 0
    fi

    echo "Creating virtual audio devices..."
    : > "$STATE_FILE"

    mic_id=$(pactl load-module module-null-sink \
        sink_name=Gabriel_Mic \
        sink_properties=device.description=Gabriel_Mic)
    echo "$mic_id" >> "$STATE_FILE"

    ears_id=$(pactl load-module module-null-sink \
        sink_name=Gabriel_Ears \
        sink_properties=device.description=Gabriel_Ears)
    echo "$ears_id" >> "$STATE_FILE"

    # so you can hear what the AI hears, mirror Gabriel_Ears to your real speakers
    loop_id=$(pactl load-module module-loopback \
        source=Gabriel_Ears.monitor latency_msec=60)
    echo "$loop_id" >> "$STATE_FILE"

    echo
    echo "Done. Two sinks are live: Gabriel_Mic and Gabriel_Ears."
    echo
    echo "VRChat side (pavucontrol makes this painless):"
    echo "  Recording: capture from 'Monitor of Gabriel_Mic'"
    echo "  Playback:  output to    'Gabriel_Ears'"
    echo
    echo "AI side:"
    echo "  PortAudio only shows 'pipewire', 'pulse' and 'default' here, not the"
    echo "  individual sinks, so you cant pick Gabriel_Mic by index. instead set"
    echo "  BOTH input_device and output_device in config.yml to the 'pulse' index"
    echo "  (see the lister below), then start with the sinks selected:"
    echo "    PULSE_SOURCE=Gabriel_Ears.monitor PULSE_SINK=Gabriel_Mic ./run.sh"
    echo "  ./run.sh already sets those for you when these sinks are loaded."
    echo
    echo "List the device indexes PortAudio sees with:"
    echo "  .venv/bin/python -c \"import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]\""
    echo
    echo "Want to hear the AI's own voice while testing (optional):"
    echo "  pactl load-module module-loopback source=Gabriel_Mic.monitor latency_msec=60"
    echo
    echo "These sinks vanish on reboot. Re-run '$0 up' after a restart."
}

cmd_down() {
    need_pactl
    if [ ! -f "$STATE_FILE" ]; then
        echo "Nothing to tear down (no state file)."
        exit 0
    fi
    while read -r id; do
        [ -n "$id" ] && pactl unload-module "$id" 2>/dev/null
    done < "$STATE_FILE"
    rm -f "$STATE_FILE"
    echo "Removed the Gabriel virtual audio devices."
}

cmd_status() {
    need_pactl
    echo "Sinks:"
    pactl list short sinks | grep -i gabriel || echo "  (none)"
    echo "Sources:"
    pactl list short sources | grep -i gabriel || echo "  (none)"
}

case "${1:-up}" in
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    *) echo "usage: $0 [up|down|status]"; exit 1 ;;
esac
