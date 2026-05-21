Shader "ProjectGabriel/PoseExfilScreen"
{
    // Screen-space pose exfil. Uses the standard fullscreen quad trick:
    // stretch any mesh to NDC corners in the vert shader, then in the frag
    // shader paint a tiny grid in the screen corner.
    //
    // Each "logical pixel" is encoded as 1 bit per RGB channel (pure 0 or
    // pure 1). Pure 0/1 round-trip through sRGB gamma cleanly, so screen
    // capture sees exactly what we wrote, no tonemapping issues.
    //
    // LAYOUT (34 cells wide x 2 cells tall, anchored bottom-left):
    //   Row 0 (bottom): position bits 0..31 in cells 0..31,
    //                   cell 32 = RED marker, cell 33 = GREEN marker
    //   Row 1 (above):  forward bits 0..31 in cells 0..31,
    //                   cell 32 = GREEN marker, cell 33 = RED marker
    //
    // Each data cell encodes 3 values at the same bit position:
    //   R = X bit p, G = Y bit p, B = Z bit p
    //
    // Float packing (matches src/pose_decoder.py):
    //   uint = (float + 5000) * 100      -> 1cm precision, +-5000m range
    //
    // The marker pixels give the decoder a fixed RGB pattern to locate the
    // strip on screen, and let it tell row 0 from row 1.
    //
    // World pose comes from unity_ObjectToWorld so this object must travel
    // with the avatar. Parenting at the avatar root works.

    Properties
    {
        _CellSize  ("Cell Size (px per logical pixel)", Float) = 8
        _OffsetX   ("Offset X (px from left)",          Float) = 0
        _OffsetY   ("Offset Y (px from bottom)",        Float) = 0
    }

    SubShader
    {
        // queue is bumped way above VRChat nameplates and other Overlay
        // shaders so the strip cant get covered by UI when people walk in
        // front of the camera. ZTest Always + ZWrite Off also helps.
        Tags { "Queue"="Overlay+5000" "RenderType"="Overlay" "IgnoreProjector"="True" "DisableBatching"="True" "PreviewType"="Plane" }
        LOD 100
        ZTest Always
        ZWrite Off
        Cull Off
        Lighting Off
        // Force opaque replace - we want pure 0/1 in each channel to land
        // in the framebuffer unchanged. no alpha blend, no nameplate color
        // bleed through, no premultiply weirdness from fallback paths.
        Blend Off
        BlendOp Add
        ColorMask RGBA
        Fog { Mode Off }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float _CellSize;
            float _OffsetX;
            float _OffsetY;

            struct appdata { float4 vertex : POSITION; };
            struct v2f    { float4 vertex : SV_POSITION; };

            // Snap every vertex to a screen corner by the sign of its x/y.
            // Works for any quad whose four corners straddle zero in object
            // space. The actual grid placement is done in the frag shader.
            v2f vert(appdata v)
            {
                v2f o;
                o.vertex.x = v.vertex.x < 0 ? -1.0 :  1.0;
                o.vertex.y = v.vertex.y < 0 ? -1.0 :  1.0;
                o.vertex.z = 1.0;
                o.vertex.w = 1.0;
                return o;
            }

            // pack a float to uint, offset by 5000 to handle negatives,
            // 1cm precision.
            uint packFloat(float f)
            {
                f += 5000.0;
                return (uint)(f * 100.0);
            }

            // write bit p of x/y/z to the R/G/B channels of one pixel.
            // each channel is either 0.0 or 1.0, which survives sRGB.
            float4 writeBits(uint x, uint y, uint z, uint p)
            {
                float4 col = float4(0, 0, 0, 1);
                if (p < 32u)
                {
                    col.r = (float)((x >> p) & 1u);
                    col.g = (float)((y >> p) & 1u);
                    col.b = (float)((z >> p) & 1u);
                }
                return col;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // SV_POSITION in the frag shader = pixel coord. Anchor the
                // grid to the bottom-left of the framebuffer. Unity flips Y
                // when rendering to intermediate RTs on some platforms, but
                // we account for that by reading the projection sign so the
                // strip ends up at the bottom of the FINAL image regardless.
                float cell = max(1.0, _CellSize);
                float px = i.vertex.x - _OffsetX;
                // _ProjectionParams.x is -1 when Y is flipped during render
                // (intermediate RTs, post effects, etc). when flipped,
                // SV_POSITION.y already measures from the bottom of the RT,
                // so we use it directly. when NOT flipped (typical desktop
                // forward path on D3D11), SV_POSITION.y is from the top
                // so we mirror with _ScreenParams.y - y.
                float yFromBottom = (_ProjectionParams.x < 0.0)
                    ? i.vertex.y
                    : (_ScreenParams.y - i.vertex.y);
                float py = yFromBottom - _OffsetY;

                int cellX = (int)floor(px / cell);
                int cellY = (int)floor(py / cell);

                // clip everything outside our 34 x 2 cell grid
                if (cellX < 0 || cellX > 33) clip(-1);
                if (cellY < 0 || cellY > 1)  clip(-1);

                if (cellY == 0)
                {
                    // position row
                    if (cellX == 32) return fixed4(1, 0, 0, 1);  // RED marker
                    if (cellX == 33) return fixed4(0, 1, 0, 1);  // GREEN marker

                    uint x = packFloat(unity_ObjectToWorld._m03);
                    uint y = packFloat(unity_ObjectToWorld._m13);
                    uint z = packFloat(unity_ObjectToWorld._m23);
                    return writeBits(x, y, z, (uint)cellX);
                }
                else
                {
                    // forward vector row (used to derive yaw)
                    if (cellX == 32) return fixed4(0, 1, 0, 1);  // GREEN marker
                    if (cellX == 33) return fixed4(1, 0, 0, 1);  // RED marker

                    uint x = packFloat(unity_ObjectToWorld._m02);
                    uint y = packFloat(unity_ObjectToWorld._m12);
                    uint z = packFloat(unity_ObjectToWorld._m22);
                    return writeBits(x, y, z, (uint)cellX);
                }
            }
            ENDCG
        }
    }
}
