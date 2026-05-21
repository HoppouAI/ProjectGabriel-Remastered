#if UNITY_EDITOR
using System;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace ProjectGabriel.Editor
{
    /// <summary>
    /// One-stop installer window. Drag your avatar in, tick what you want,
    /// hit Install. The window builds the prefabs (if not already built)
    /// and parents fresh instances onto the avatar at identity transforms.
    ///
    /// Menu: Tools > ProjectGabriel > Avatar Setup (Installer)
    ///
    /// The older "Build Pose HUD" / "Build Sensor Rig" menu items still
    /// work and just dump the prefabs into the project without touching a
    /// scene avatar, in case you prefer doing the drag-on step yourself.
    /// </summary>
    public class GabrielAvatarSetupWindow : EditorWindow
    {
        private GameObject _avatar;
        private bool _installPoseHud = true;
        private bool _installSensorRig = true;
        private bool _replaceExisting = true;
        private Vector2 _scroll;
        private string _statusLog;

        [MenuItem("Tools/ProjectGabriel/Avatar Setup (Installer)")]
        public static void ShowWindow()
        {
            var w = GetWindow<GabrielAvatarSetupWindow>(false, "Gabriel Avatar Setup", true);
            w.minSize = new Vector2(440, 460);
            w.Show();
        }

        private void OnEnable()
        {
            // grab whatever's selected so the user doesn't have to drag
            // every time they open the window
            if (_avatar == null && Selection.activeGameObject != null)
            {
                var candidate = FindAvatarRoot(Selection.activeGameObject);
                if (candidate != null) _avatar = candidate;
            }
        }

        private void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            DrawHeader();
            EditorGUILayout.Space(8);
            DrawAvatarPicker();
            EditorGUILayout.Space(8);
            DrawInstallOptions();
            EditorGUILayout.Space(12);
            DrawInstallButton();
            EditorGUILayout.Space(12);
            DrawStatus();
            EditorGUILayout.Space(8);
            DrawFooterLinks();

            EditorGUILayout.EndScrollView();
        }

        // --- sections ---------------------------------------------------------

        private void DrawHeader()
        {
            var style = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 16,
                alignment = TextAnchor.MiddleLeft
            };
            GUILayout.Label("ProjectGabriel Avatar Setup", style);
            EditorGUILayout.LabelField(
                "Drag your avatar in, pick what to install, hit Install.",
                EditorStyles.miniLabel);
        }

        private void DrawAvatarPicker()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("1. Target avatar", EditorStyles.boldLabel);

            EditorGUI.BeginChangeCheck();
            var picked = (GameObject)EditorGUILayout.ObjectField(
                new GUIContent("Avatar (scene object)",
                    "Drag the GameObject that has the VRCAvatarDescriptor."),
                _avatar, typeof(GameObject), true);
            if (EditorGUI.EndChangeCheck())
            {
                _avatar = picked != null ? FindAvatarRoot(picked) : null;
            }

            if (_avatar == null)
            {
                EditorGUILayout.HelpBox(
                    "No avatar selected. Drag your avatar from the Hierarchy here, " +
                    "or click 'Find avatar in scene' below.",
                    MessageType.Info);
            }
            else
            {
                var hasDescriptor = HasAvatarDescriptor(_avatar);
                EditorGUILayout.HelpBox(
                    hasDescriptor
                        ? "Detected VRCAvatarDescriptor on '" + _avatar.name + "'. Looks good."
                        : "'" + _avatar.name + "' does not have a VRCAvatarDescriptor. " +
                          "You can still install but make sure this is the avatar root.",
                    hasDescriptor ? MessageType.Info : MessageType.Warning);
            }

            if (GUILayout.Button("Find avatar in scene"))
            {
                var found = FindFirstAvatarInScene();
                if (found != null)
                {
                    _avatar = found;
                    Log("auto-detected avatar: " + found.name);
                }
                else
                {
                    Log("no VRCAvatarDescriptor found in the open scene.");
                }
            }

            EditorGUILayout.EndVertical();
        }

        private void DrawInstallOptions()
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);
            EditorGUILayout.LabelField("2. What to install", EditorStyles.boldLabel);

            _installPoseHud = EditorGUILayout.ToggleLeft(
                new GUIContent("Pose HUD (bottom-left coord strip)",
                    "Adds the GabrielPoseHUD child that broadcasts world XYZ + yaw via a screen-space color strip."),
                _installPoseHud);

            _installSensorRig = EditorGUILayout.ToggleLeft(
                new GUIContent("Sensor Rig (raycast empties)",
                    "Adds the GabrielSensorRig child with 11 VRCRaycast empties grouped under HeadAnchor / HipsAnchor."),
                _installSensorRig);

            EditorGUILayout.Space(4);
            _replaceExisting = EditorGUILayout.ToggleLeft(
                new GUIContent("Replace existing instances",
                    "If an old GabrielPoseHUD / GabrielSensorRig child already exists on the avatar, delete it before installing the fresh one."),
                _replaceExisting);

            EditorGUILayout.EndVertical();
        }

        private void DrawInstallButton()
        {
            using (new EditorGUI.DisabledScope(_avatar == null || (!_installPoseHud && !_installSensorRig)))
            {
                var label = _avatar != null
                    ? "Install on '" + _avatar.name + "'"
                    : "Install (pick an avatar first)";
                if (GUILayout.Button(label, GUILayout.Height(36)))
                {
                    RunInstall();
                }
            }
        }

        private void DrawStatus()
        {
            if (string.IsNullOrEmpty(_statusLog)) return;
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            EditorGUILayout.TextArea(_statusLog, GUILayout.MinHeight(80));
        }

        private void DrawFooterLinks()
        {
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open setup guide (AVATAR_SETUP.md)"))
            {
                var guesses = new[]
                {
                    "Assets/ProjectGabriel/AVATAR_SETUP.md",
                    "Assets/ProjectGabriel/unity_assets/AVATAR_SETUP.md"
                };
                foreach (var g in guesses)
                {
                    var obj = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(g);
                    if (obj != null) { AssetDatabase.OpenAsset(obj); return; }
                }
                Log("AVATAR_SETUP.md not found under Assets/ProjectGabriel. Check the repo's unity_assets/ folder.");
            }
            if (GUILayout.Button("Rebuild prefabs only (no install)"))
            {
                GabrielPoseHudBuilder.BuildPrefab(verbose: true);
                GabrielSensorRigBuilder.BuildPrefab(verbose: true);
                Log("rebuilt both prefabs under Assets/ProjectGabriel/Generated/.");
            }
            EditorGUILayout.EndHorizontal();
        }

        // --- install pipeline -------------------------------------------------

        private void RunInstall()
        {
            _statusLog = "";
            if (_avatar == null) { Log("no avatar selected."); return; }

            Undo.RegisterFullObjectHierarchyUndo(_avatar, "Install Gabriel Sensing");

            if (_installPoseHud)
            {
                var path = GabrielPoseHudBuilder.BuildPrefab(verbose: false);
                if (string.IsNullOrEmpty(path)) Log("pose HUD: build failed (see console).");
                else InstallPrefab(path, _avatar, "GabrielPoseHUD");
            }

            if (_installSensorRig)
            {
                var path = GabrielSensorRigBuilder.BuildPrefab(verbose: false);
                if (string.IsNullOrEmpty(path)) Log("sensor rig: build failed (see console).");
                else InstallPrefab(path, _avatar, "GabrielSensorRig");
            }

            EditorSceneSetDirty(_avatar);
            Log("done. select the avatar in the Hierarchy to see the new children.");
            EditorGUIUtility.PingObject(_avatar);
            Selection.activeGameObject = _avatar;
        }

        private void InstallPrefab(string prefabPath, GameObject avatar, string expectedName)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) { Log(expectedName + ": prefab missing at " + prefabPath); return; }

            if (_replaceExisting)
            {
                var existing = FindChildByName(avatar.transform, expectedName);
                if (existing != null)
                {
                    Undo.DestroyObjectImmediate(existing.gameObject);
                    Log(expectedName + ": removed existing instance.");
                }
            }

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, avatar.transform);
            if (instance == null) { Log(expectedName + ": instantiate returned null."); return; }
            Undo.RegisterCreatedObjectUndo(instance, "Install " + expectedName);
            instance.transform.localPosition = Vector3.zero;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale    = Vector3.one;
            Log(expectedName + ": installed as a child of '" + avatar.name + "'.");
        }

        // --- helpers ----------------------------------------------------------

        private static GameObject FindAvatarRoot(GameObject picked)
        {
            // walk up to the first ancestor with a VRCAvatarDescriptor. fall
            // back to the dragged object itself so the user can still pick a
            // bare GameObject (e.g. for testing without the SDK).
            var t = picked.transform;
            while (t != null)
            {
                if (HasAvatarDescriptor(t.gameObject)) return t.gameObject;
                t = t.parent;
            }
            return picked;
        }

        private static bool HasAvatarDescriptor(GameObject go)
        {
            var t = FindType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor")
                    ?? FindType("VRC.SDKBase.VRC_AvatarDescriptor");
            if (t == null) return false;
            return go.GetComponent(t) != null;
        }

        private static GameObject FindFirstAvatarInScene()
        {
            var t = FindType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor")
                    ?? FindType("VRC.SDKBase.VRC_AvatarDescriptor");
            if (t == null) return null;
            var descriptors = UnityEngine.Object.FindObjectsByType(t,
                FindObjectsInactive.Include, FindObjectsSortMode.None);
            if (descriptors == null || descriptors.Length == 0) return null;
            var comp = descriptors[0] as Component;
            return comp != null ? comp.gameObject : null;
        }

        private static Type FindType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(fullName, false);
                if (t != null) return t;
            }
            return null;
        }

        private static Transform FindChildByName(Transform parent, string name)
        {
            for (int i = 0; i < parent.childCount; i++)
            {
                var c = parent.GetChild(i);
                if (c.name == name) return c;
            }
            return null;
        }

        private static void EditorSceneSetDirty(GameObject go)
        {
            // make sure the change actually saves with the scene
            var scene = go.scene;
            if (scene.IsValid())
            {
                UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(scene);
            }
        }

        private void Log(string line)
        {
            if (string.IsNullOrEmpty(_statusLog)) _statusLog = line;
            else _statusLog += "\n" + line;
            Repaint();
        }
    }
}
#endif
