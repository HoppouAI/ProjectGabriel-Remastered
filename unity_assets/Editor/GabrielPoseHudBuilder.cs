#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace ProjectGabriel.Editor
{
    /// <summary>
    /// Builds the "Gabriel Pose HUD" prefab using the screen-space PoseExfil
    /// shader.
    ///
    /// Setup: drop the prefab onto your avatar root and upload. Do NOT
    /// parent it under the Head bone. VRChat scales the local player's
    /// head bone to zero in first person to hide head-mounted props, so
    /// anything under Head goes invisible to yourself. This prefab sits at
    /// the avatar root and uses a tall mesh bounding box (covers feet to
    /// above head) so the camera is always inside the renderer AABB
    /// regardless of first/third person, leaning, walking, or looking up.
    ///
    /// The vertex shader uses sign(vertex.xy) to snap to NDC corners so the
    /// mesh size only affects culling, not what gets drawn. The strip
    /// always lands in the bottom-left of the screen.
    ///
    /// Menu: Tools > ProjectGabriel > Build Pose HUD
    ///
    /// PRIVACY WARNING: anyone whose camera renders your avatar will also
    /// see the cell grid in their view, and a determined attacker could
    /// decode your coords. Keep the cell size small and turn off the
    /// renderer when you don't need navigation.
    /// Default grid is 34x2 cells at 8 px per cell = 272x16 screen pixels.
    /// </summary>
    public static class GabrielPoseHudBuilder
    {
        internal const string OUTPUT_DIR = "Assets/ProjectGabriel/Generated";
        internal const string PREFAB_NAME = "GabrielPoseHUD.prefab";
        private const string MAT_NAME = "GabrielPoseHUD.mat";
        private const string MESH_NAME = "GabrielPoseHUDQuad.asset";
        private const string SHADER_NAME = "ProjectGabriel/PoseExfilScreen";

        // grid is GRID_W x GRID_H cells, each CELL_SIZE x CELL_SIZE physical
        // pixels. must match the constants in src/pose_decoder.py.
        private const float CELL_SIZE = 4f;
        private const float OFFSET_X  = 0f;    // pixels from left edge
        private const float OFFSET_Y  = 0f;    // pixels from bottom edge

        // mesh AABB in object space. centered roughly at chest height so
        // the box covers from slightly below the feet (-1) to slightly
        // above the head (+2.5) and ~1.5m to each side. avatar AABB stays
        // well under the 5m VRChat limit on every axis.
        private static readonly Vector3 BOUNDS_CENTER = new Vector3(0f, 0.75f, 0f);
        private static readonly Vector3 BOUNDS_SIZE   = new Vector3(3f, 3.5f, 3f);

        [MenuItem("Tools/ProjectGabriel/Build Pose HUD")]
        public static void Build()
        {
            var path = BuildPrefab(verbose: true);
            if (string.IsNullOrEmpty(path)) return;
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            Selection.activeObject = prefab;
            EditorGUIUtility.PingObject(prefab);
        }

        /// <summary>
        /// Rebuilds the prefab on disk and returns its asset path. Safe to
        /// call from other editor scripts (the installer window uses this).
        /// Returns null if the shader couldn't be found.
        /// </summary>
        public static string BuildPrefab(bool verbose)
        {
            EnsureFolder(OUTPUT_DIR);

            var shader = Shader.Find(SHADER_NAME);
            if (shader == null)
            {
                if (verbose)
                {
                    EditorUtility.DisplayDialog(
                        "Pose HUD",
                        "Could not find shader '" + SHADER_NAME + "'. Make sure " +
                        "unity_assets/shaders/PoseExfilScreen.shader is imported " +
                        "into the Unity project first.",
                        "OK");
                }
                else
                {
                    Debug.LogError("[GabrielPoseHudBuilder] shader not found: " + SHADER_NAME);
                }
                return null;
            }

            var matPath = OUTPUT_DIR + "/" + MAT_NAME;
            var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
            if (mat == null)
            {
                mat = new Material(shader);
                mat.name = Path.GetFileNameWithoutExtension(MAT_NAME);
                AssetDatabase.CreateAsset(mat, matPath);
            }
            else if (mat.shader != shader)
            {
                mat.shader = shader;
            }
            mat.SetFloat("_CellSize", CELL_SIZE);
            mat.SetFloat("_OffsetX",  OFFSET_X);
            mat.SetFloat("_OffsetY",  OFFSET_Y);
            EditorUtility.SetDirty(mat);

            var root = new GameObject("GabrielPoseHUD");
            // ship the prefab with the root disabled. the VRCFury toggle
            // (Default On + In Local: turn on GabrielPoseHUD) is what
            // wakes it up at world join. if you skip the toggle, just
            // tick the GameObject active in the inspector before upload.
            root.SetActive(false);
            try
            {
                var meshPath = OUTPUT_DIR + "/" + MESH_NAME;
                var quadMesh = AssetDatabase.LoadAssetAtPath<Mesh>(meshPath);
                if (quadMesh == null)
                {
                    quadMesh = BuildQuadMesh();
                    AssetDatabase.CreateAsset(quadMesh, meshPath);
                }
                else
                {
                    // refresh cached mesh so old builds (skinned, tiny bounds,
                    // huge bounds, whatever) get replaced with current verts.
                    var fresh = BuildQuadMesh();
                    quadMesh.Clear();
                    quadMesh.vertices = fresh.vertices;
                    quadMesh.uv = fresh.uv;
                    quadMesh.triangles = fresh.triangles;
                    quadMesh.boneWeights = new BoneWeight[0];
                    quadMesh.bindposes = new Matrix4x4[0];
                    quadMesh.RecalculateNormals();
                    quadMesh.bounds = fresh.bounds;
                    EditorUtility.SetDirty(quadMesh);
                }

                var quad = new GameObject("PoseStrip");
                quad.transform.SetParent(root.transform, false);
                quad.transform.localPosition = Vector3.zero;
                quad.transform.localRotation = Quaternion.identity;
                quad.transform.localScale = Vector3.one;

                var mf = quad.AddComponent<MeshFilter>();
                mf.sharedMesh = quadMesh;
                var renderer = quad.AddComponent<MeshRenderer>();
                renderer.sharedMaterial = mat;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                renderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
                renderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
                renderer.allowOcclusionWhenDynamic = false;

                var prefabPath = OUTPUT_DIR + "/" + PREFAB_NAME;
                PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
                AssetDatabase.SaveAssets();
                if (verbose)
                {
                    Debug.Log("[GabrielPoseHudBuilder] built " + prefabPath +
                              ". Drop it on your avatar ROOT (NOT under the " +
                              "Head bone - VRChat hides head children in first " +
                              "person) and upload. A 34x2 pixel strip will " +
                              "appear in the bottom-left corner encoding world " +
                              "XYZ + yaw.");
                }
                return prefabPath;
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            var parts = path.Split('/');
            var cur = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                var next = cur + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                {
                    AssetDatabase.CreateFolder(cur, parts[i]);
                }
                cur = next;
            }
        }

        private static Mesh BuildQuadMesh()
        {
            // verts are symmetric +-0.5 in object space, but the bounds
            // override below makes the cull AABB span feet-to-above-head
            // so the camera is always inside regardless of perspective.
            var mesh = new Mesh();
            mesh.name = "GabrielPoseHUDQuad";
            mesh.vertices = new Vector3[]
            {
                new Vector3(-0.5f, -0.5f, 0f),
                new Vector3( 0.5f, -0.5f, 0f),
                new Vector3(-0.5f,  0.5f, 0f),
                new Vector3( 0.5f,  0.5f, 0f),
            };
            mesh.uv = new Vector2[]
            {
                new Vector2(0f, 0f),
                new Vector2(1f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
            };
            mesh.triangles = new int[] { 0, 2, 1, 2, 3, 1 };
            mesh.RecalculateNormals();
            mesh.bounds = new Bounds(BOUNDS_CENTER, BOUNDS_SIZE);
            return mesh;
        }
    }
}
#endif
