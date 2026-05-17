Shader "ProjectGabriel/PoseExfil"
{
    // Renders an 8x1 pixel strip encoding the avatar's world position + yaw.
    // Python screen-captures the strip (default top-left of screen) and
    // decodes it back to a WorldPose. See src/pose_decoder.py for the spec.
    //
    // PIXEL LAYOUT (RGB bytes 0..255):
    //   0: magic           (0xDE, 0xAD, 0xBE)
    //   1: version + flags (0x01, 0, 0)
    //   2: X cm signed 24  (R=high, G=mid, B=low)   two's complement
    //   3: Y cm signed 24
    //   4: Z cm signed 24
    //   5: yaw 16-bit      (R=high, G=low, B=0)     0..65535 -> 0..360
    //   6: reserved        (0, 0, 0)
    //   7: checksum        (R = XOR of all preceding bytes, G=0, B=0)
    //
    // USAGE:
    //   1. Make a small unlit Quad in your avatar (1 unit wide, 0.125 tall is fine).
    //   2. Parent it under your camera/head so it's always on screen.
    //   3. Position so it covers exactly 8 screen pixels horizontally, 1 vertical.
    //      Easiest setup: use a Canvas in Screen Space - Camera with a RawImage
    //      8x1 px at (0,0) and assign this material.
    //   4. ZTest Always + ZWrite Off so nothing occludes it.
    //   5. Make sure it's NOT visible to other players (use a Camera-only layer
    //      or render only on local camera) so you don't broadcast your coords.
    //
    // SAFETY: shader exposes world coords to anyone running OBS on your screen.
    // Keep the strip in a corner you can crop out for streams.

    Properties
    {
        _StripWidthPixels ("Strip Width (px)", Float) = 8
    }

    SubShader
    {
        Tags { "Queue"="Overlay" "RenderType"="Opaque" "IgnoreProjector"="True" }
        ZTest Always
        ZWrite Off
        Cull Off
        Lighting Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float _StripWidthPixels;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f    { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv  = v.uv;
                return o;
            }

            // pack a signed int (24-bit range) into an RGB triple, two's complement
            float3 PackSigned24(int v)
            {
                int u = v;
                if (u < 0) u += 16777216; // 2^24
                int r = (u >> 16) & 0xFF;
                int g = (u >> 8)  & 0xFF;
                int b =  u        & 0xFF;
                return float3(r, g, b) / 255.0;
            }

            float3 PackUnsigned16(int v)
            {
                int u = v & 0xFFFF;
                int r = (u >> 8) & 0xFF;
                int g =  u       & 0xFF;
                return float3(r, g, 0) / 255.0;
            }

            int XorByte(int a, int b) { return (a ^ b) & 0xFF; }

            int ByteFromFloat(float c) { return (int)round(saturate(c) * 255.0); }

            fixed4 frag(v2f i) : SV_Target
            {
                // which pixel of the strip are we drawing? uv.x in [0,1)
                int pixelIndex = (int)floor(i.uv.x * _StripWidthPixels);
                pixelIndex = clamp(pixelIndex, 0, 7);

                // world position from this object's transform matrix
                // _m03,_m13,_m23 is the translation column
                float3 worldPos = float3(
                    unity_ObjectToWorld._m03,
                    unity_ObjectToWorld._m13,
                    unity_ObjectToWorld._m23
                );

                // yaw from the object-to-world rotation. Unity yaw measured
                // clockwise from +Z when looking down at XZ plane.
                // forward vector = ObjectToWorld * (0,0,1,0)
                float3 fwd = float3(
                    unity_ObjectToWorld._m02,
                    unity_ObjectToWorld._m12,
                    unity_ObjectToWorld._m22
                );
                // yaw in degrees, 0..360
                float yawDeg = degrees(atan2(fwd.x, fwd.z));
                if (yawDeg < 0) yawDeg += 360.0;

                // pack values
                int xCm   = (int)round(worldPos.x * 100.0);
                int yCm   = (int)round(worldPos.y * 100.0);
                int zCm   = (int)round(worldPos.z * 100.0);
                int yawU  = (int)round(yawDeg * (65536.0 / 360.0)) & 0xFFFF;

                float3 magic    = float3(0xDE, 0xAD, 0xBE) / 255.0;
                float3 version  = float3(0x01, 0, 0) / 255.0;
                float3 xPix     = PackSigned24(xCm);
                float3 yPix     = PackSigned24(yCm);
                float3 zPix     = PackSigned24(zCm);
                float3 yawPix   = PackUnsigned16(yawU);
                float3 reserved = float3(0, 0, 0);

                // checksum = XOR of all bytes in pixels 0..6
                int cs = 0;
                cs = XorByte(cs, ByteFromFloat(magic.r));
                cs = XorByte(cs, ByteFromFloat(magic.g));
                cs = XorByte(cs, ByteFromFloat(magic.b));
                cs = XorByte(cs, ByteFromFloat(version.r));
                cs = XorByte(cs, ByteFromFloat(version.g));
                cs = XorByte(cs, ByteFromFloat(version.b));
                cs = XorByte(cs, ByteFromFloat(xPix.r));
                cs = XorByte(cs, ByteFromFloat(xPix.g));
                cs = XorByte(cs, ByteFromFloat(xPix.b));
                cs = XorByte(cs, ByteFromFloat(yPix.r));
                cs = XorByte(cs, ByteFromFloat(yPix.g));
                cs = XorByte(cs, ByteFromFloat(yPix.b));
                cs = XorByte(cs, ByteFromFloat(zPix.r));
                cs = XorByte(cs, ByteFromFloat(zPix.g));
                cs = XorByte(cs, ByteFromFloat(zPix.b));
                cs = XorByte(cs, ByteFromFloat(yawPix.r));
                cs = XorByte(cs, ByteFromFloat(yawPix.g));
                cs = XorByte(cs, ByteFromFloat(yawPix.b));
                cs = XorByte(cs, ByteFromFloat(reserved.r));
                cs = XorByte(cs, ByteFromFloat(reserved.g));
                cs = XorByte(cs, ByteFromFloat(reserved.b));
                float3 checksum = float3(cs, 0, 0) / 255.0;

                float3 outRGB;
                if      (pixelIndex == 0) outRGB = magic;
                else if (pixelIndex == 1) outRGB = version;
                else if (pixelIndex == 2) outRGB = xPix;
                else if (pixelIndex == 3) outRGB = yPix;
                else if (pixelIndex == 4) outRGB = zPix;
                else if (pixelIndex == 5) outRGB = yawPix;
                else if (pixelIndex == 6) outRGB = reserved;
                else                       outRGB = checksum;

                return fixed4(outRGB, 1.0);
            }
            ENDCG
        }
    }
}
