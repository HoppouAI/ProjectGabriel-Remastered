{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  x11-libs = with pkgs.xorg; [
    libX11
    libXext
    libXcursor
    libXrender
    libXi
    libXrandr
    libXfixes
    libXinerama
    libXxf86vm
    libSM
    libICE
  ];
  # native libs exposed on LD_LIBRARY_PATH at runtime + build time
  native-libs =
    with pkgs;
    [
      portaudio
      alsa-lib
      pipewire
      pulseaudio
      libsndfile
      ffmpeg
      mesa
      libglvnd
      glib
      zlib
    ]
    ++ x11-libs;
in

pkgs.mkShell {
  buildInputs = [
    pkgs.git
    pkgs.pkg-config
    pkgs.linuxHeaders
  ]
  ++ native-libs;

  shellHook = ''
    # make the native .so files visible to the Python process at runtime and to
    # uv's build subprocess at build time (pyaudio, soundfile, opencv, pygame).
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath native-libs}:$LD_LIBRARY_PATH"

    # CUDA runtime libs bundled by the torch cu126 pip wheels live under
    # .venv/.../nvidia/*/lib and arent on the default linker path in a nix shell.
    # no-op for the CPU build (find returns nothing).
    NVIDIA_LIBS="$(find .venv/lib/python*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')"
    if [ -n "$NVIDIA_LIBS" ]; then
      export LD_LIBRARY_PATH="$NVIDIA_LIBS$LD_LIBRARY_PATH"
    fi

    # *.pc dirs so build-time pkg-config finds portaudio / alsa / pulseaudio.
    export PKG_CONFIG_PATH="${pkgs.portaudio}/lib/pkgconfig:${pkgs.alsa-lib}/lib/pkgconfig:${pkgs.pipewire}/lib/pkgconfig:${pkgs.pulseaudio}/lib/pkgconfig:${pkgs.libsndfile}/lib/pkgconfig:$PKG_CONFIG_PATH"

    # C headers for any wheel that compiles a C extension.
    export CFLAGS="-I${pkgs.linuxHeaders}/include -I${pkgs.portaudio}/include -I${pkgs.alsa-lib}/include -I${pkgs.pipewire}/include -I${pkgs.pulseaudio}/include -I${pkgs.libsndfile}/include -I${pkgs.glib}/include $CFLAGS"

    # evdev (a transitive dep of pynput) locates linux/input.h by reading the
    # C_INCLUDE_PATH / CPATH env vars, not CFLAGS, so expose the kernel headers
    # there too. the C compiler also honors C_INCLUDE_PATH when compiling
    # evdev's input.c / uinput.c / ecodes.c.
    export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include:$C_INCLUDE_PATH"
    export CPATH="${pkgs.linuxHeaders}/include:$CPATH"

    # imageio-ffmpeg ships a bundled ffmpeg binary that may not run in a nix
    # shell (its ELF interpreter points at /lib64/ld-linux-x86-64.so.2, which
    # isnt always present). point it at the nix ffmpeg on PATH instead.
    export IMAGEIO_FFMPEG_EXE="${pkgs.ffmpeg}/bin/ffmpeg"

    echo "=> ProjectGabriel nix shell"
  '';
}
