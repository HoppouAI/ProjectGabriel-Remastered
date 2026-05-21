#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace ProjectGabriel.Editor
{
    /// <summary>
    /// Builds a "Gabriel Sensor Rig" prefab containing all the raycast empties
    /// the python side expects, grouped under per-bone anchor children so you
    /// can drop a VRCFury Armature Link onto each anchor by hand.
    ///
    /// Menu: Tools > ProjectGabriel > Build Sensor Rig
    ///
    /// REQUIREMENTS:
    ///   - VRChat Avatar SDK3 (for VRCRaycast + VRCExpressionParameters)
    ///
    /// The script adds VRCRaycast components via reflection so it compiles
    /// even if the SDK is missing, you just get a warning.
    /// </summary>
    public static class GabrielSensorRigBuilder
    {
        internal const string OUTPUT_DIR = "Assets/ProjectGabriel/Generated";
        internal const string PREFAB_NAME = "GabrielSensorRig.prefab";
        private const string PARAMS_NAME = "GabrielSensorParameters.asset";

        // ray definition table - mirrors unity_assets/AVATAR_SETUP.md
        // (parent bone name, ray name, local euler rot, length meters)
        private struct RayDef
        {
            public string Bone;
            public string Name;
            public Vector3 LocalEuler;  // applied to the empty so its forward = ray dir
            public float Length;
            public bool HitPlayers;     // false = worlds only, true = both

            public RayDef(string bone, string name, Vector3 euler, float length, bool hitPlayers = false)
            {
                Bone = bone; Name = name; LocalEuler = euler; Length = length; HitPlayers = hitPlayers;
            }
        }

        private static readonly RayDef[] RAYS = new RayDef[]
        {
            // head-mounted nav + awareness
            new RayDef("Head", "Fwd",      new Vector3(0,   0,   0), 5f),
            new RayDef("Head", "Left",     new Vector3(0, -90,   0), 4f),
            new RayDef("Head", "Right",    new Vector3(0,  90,   0), 4f),
            new RayDef("Head", "LeftFwd",  new Vector3(0, -45,   0), 5f),
            new RayDef("Head", "RightFwd", new Vector3(0,  45,   0), 5f),
            new RayDef("Head", "Back",     new Vector3(0, 180,   0), 3f),
            new RayDef("Head", "Up",       new Vector3(-90, 0,   0), 3f),
            new RayDef("Head", "Gaze",     new Vector3(0,   0,   0), 30f, hitPlayers: true),

            // hip mounted for ledge / floor / tight nav
            new RayDef("Hips", "FwdNear",  new Vector3(0,   0,   0), 1.5f),
            new RayDef("Hips", "DropFwd",  new Vector3(45,  0,   0), 3f),
            new RayDef("Hips", "Down",     new Vector3(90,  0,   0), 2f),
        };

        [MenuItem("Tools/ProjectGabriel/Build Sensor Rig")]
        public static void Build()
        {
            var path = BuildPrefab(verbose: true);
            if (string.IsNullOrEmpty(path)) return;
            Selection.activeObject = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            EditorGUIUtility.PingObject(Selection.activeObject);
        }

        /// <summary>
        /// Rebuilds the prefab on disk and returns its asset path. The
        /// installer window calls this directly so it can drop the prefab
        /// onto an avatar in one click.
        /// </summary>
        public static string BuildPrefab(bool verbose)
        {
            EnsureFolder(OUTPUT_DIR);

            var root = new GameObject("GabrielSensorRig");
            try
            {
                // group rays by bone, each group gets its own armature-linked child
                var byBone = new Dictionary<string, List<RayDef>>();
                foreach (var ray in RAYS)
                {
                    if (!byBone.TryGetValue(ray.Bone, out var list))
                    {
                        list = new List<RayDef>();
                        byBone[ray.Bone] = list;
                    }
                    list.Add(ray);
                }

                foreach (var kv in byBone)
                {
                    var boneName = kv.Key;
                    var anchor = new GameObject(boneName + "Anchor");
                    anchor.transform.SetParent(root.transform, false);

                    // try to attach the Armature Link now so the user doesnt
                    // have to. if VRCFury isnt installed we fall back to the
                    // manual instructions in AVATAR_SETUP.md.
                    TryAddVRCFuryArmatureLink(anchor, boneName);

                    foreach (var ray in kv.Value)
                    {
                        var rayGo = new GameObject("Ray_" + ray.Name);
                        rayGo.transform.SetParent(anchor.transform, false);
                        rayGo.transform.localPosition = Vector3.zero;
                        rayGo.transform.localEulerAngles = ray.LocalEuler;

                        if (!TryAddVRCRaycast(rayGo, ray))
                        {
                            Debug.LogWarning(
                                "VRCRaycast component type not found, skipping component on " +
                                rayGo.name + ". Check VRChat SDK version (raycasts need 3.x)."
                            );
                        }
                    }
                }

                // expression parameters asset
                var paramsAsset = BuildExpressionParameters();
                string paramsPath = null;
                if (paramsAsset != null)
                {
                    paramsPath = Path.Combine(OUTPUT_DIR, PARAMS_NAME).Replace('\\', '/');
                    AssetDatabase.CreateAsset(paramsAsset, paramsPath);
                    Debug.Log("Wrote " + paramsPath);

                    // VRChat only publishes animator params over OSC if they're
                    // also in the avatar's VRCExpressionParameters. without a
                    // Full Controller merging our params asset in, the raycast
                    // values exist on the component but never hit /avatar/parameters.
                    if (!TryAddVRCFuryFullController(root, paramsAsset))
                    {
                        Debug.LogWarning(
                            "VRCFury not detected, skipping auto Full Controller wiring. " +
                            "Add a VRC Fury > Full Controller to the GabrielSensorRig root and drop " +
                            paramsPath + " into the Parameters list yourself, or the ray " +
                            "params wont publish over OSC."
                        );
                    }
                }

                // save prefab
                var prefabPath = Path.Combine(OUTPUT_DIR, PREFAB_NAME).Replace('\\', '/');
                PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                if (verbose)
                {
                    Debug.Log(
                        "Built sensor rig: " + prefabPath + ". Drag it into your avatar root, " +
                        "then add a VRCFury Armature Link to each *Anchor child (Head -> Head bone, " +
                        "Hips -> Hips bone)."
                    );
                }
                return prefabPath;
            }
            finally
            {
                GameObject.DestroyImmediate(root);
            }
        }

        // --- VRCRaycast wiring (reflection) ------------------------------------

        private static bool TryAddVRCRaycast(GameObject host, RayDef ray)
        {
            var raycastType = FindType("VRC.SDK3.Avatars.Components.VRCRaycast")
                              ?? FindType("VRC.SDKBase.VRCRaycast")
                              ?? FindType("VRCRaycast");
            if (raycastType == null) return false;

            var component = host.AddComponent(raycastType);
            if (component == null) return false;

            // VRC Raycast inspector fields (verified visually): parameter (string),
            // direction (Vector3, local to this transform), distance (float),
            // applyTransformScale (bool), collisionMode (enum), resultTransform
            // (Transform, REQUIRED or the component does nothing), applyRotation
            // (bool), behaviorOnMiss (enum). The empty's own euler rotation
            // already points the desired way, so direction stays (0,0,1).
            SetField(component, "parameter", ray.Name);
            SetField(component, "direction", Vector3.forward);
            TrySetFloat(component, new[] { "distance", "length", "maxLength", "rayLength" }, ray.Length);
            TrySetEnum(component, new[] { "collisionMode", "hitMode" },
                       ray.HitPlayers ? "HitBoth" : "HitWorlds",
                       fallbackInts: new[] { ray.HitPlayers ? 2 : 0 });
            TrySetEnum(component, new[] { "behaviorOnMiss", "missBehavior" },
                       "SnapToEnd", fallbackInts: new[] { 2 });

            // result transform is mandatory, create a child empty for the hit point
            var resultGo = new GameObject("Result");
            resultGo.transform.SetParent(host.transform, false);
            resultGo.transform.localPosition = Vector3.zero;
            resultGo.transform.localRotation = Quaternion.identity;
            SetField(component, "resultTransform", resultGo.transform);

            return true;
        }

        // --- VRCFury Armature Link wiring (reflection, optional) ---------------

        // Uses VRCFury's public API to drop an Armature Link onto the anchor
        // pointed at the matching humanoid bone, with Align Position/Rotation/Scale
        // turned on. without alignment the anchor stays at world 0 after the
        // reparent and the rays fire from nowhere useful.
        private static bool TryAddVRCFuryArmatureLink(GameObject anchor, string boneName)
        {
            if (!Enum.IsDefined(typeof(HumanBodyBones), boneName))
            {
                Debug.LogWarning("No HumanBodyBones value matches anchor '" + boneName + "', skipping Armature Link.");
                return false;
            }
            var humanoidBone = (HumanBodyBones)Enum.Parse(typeof(HumanBodyBones), boneName);

            var furyComponents = FindType("com.vrcfury.api.FuryComponents");
            if (furyComponents == null) return false;

            var create = furyComponents.GetMethod(
                "CreateArmatureLink",
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(GameObject) },
                null
            );
            if (create == null) return false;

            object fal;
            try { fal = create.Invoke(null, new object[] { anchor }); }
            catch (Exception e) { Debug.LogWarning("VRCFury CreateArmatureLink threw: " + e.Message); return false; }
            if (fal == null) return false;

            var linkTo = fal.GetType().GetMethod("LinkTo", new[] { typeof(HumanBodyBones), typeof(string) });
            var setAlign = fal.GetType().GetMethod("SetAlign", new[] { typeof(bool) });
            if (linkTo == null || setAlign == null) return false;

            try
            {
                linkTo.Invoke(fal, new object[] { humanoidBone, "" });
                setAlign.Invoke(fal, new object[] { true });
            }
            catch (Exception e)
            {
                Debug.LogWarning("VRCFury Armature Link config threw: " + e.Message);
                return false;
            }

            Debug.Log("Attached VRCFury Armature Link on " + anchor.name + " -> " + boneName + " (aligned).");
            return true;
        }

        // --- VRCFury Armature Link wiring (reflection, optional) ---------------

        // Uses VRCFury's public API to drop an Armature Link onto the anchor
        // pointed at the matching humanoid bone, with Align Position/Rotation/Scale
        // turned on. without alignment the anchor stays at world 0 after the
        // reparent and the rays fire from nowhere useful.
        private static bool TryAddVRCFuryArmatureLink(GameObject anchor, string boneName)
        {
            if (!Enum.IsDefined(typeof(HumanBodyBones), boneName))
            {
                Debug.LogWarning("No HumanBodyBones value matches anchor '" + boneName + "', skipping Armature Link.");
                return false;
            }
            var humanoidBone = (HumanBodyBones)Enum.Parse(typeof(HumanBodyBones), boneName);

            var furyComponents = FindType("com.vrcfury.api.FuryComponents");
            if (furyComponents == null) return false;

            var create = furyComponents.GetMethod(
                "CreateArmatureLink",
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(GameObject) },
                null
            );
            if (create == null) return false;

            object fal;
            try { fal = create.Invoke(null, new object[] { anchor }); }
            catch (Exception e) { Debug.LogWarning("VRCFury CreateArmatureLink threw: " + e.Message); return false; }
            if (fal == null) return false;

            var linkTo = fal.GetType().GetMethod("LinkTo", new[] { typeof(HumanBodyBones), typeof(string) });
            var setAlign = fal.GetType().GetMethod("SetAlign", new[] { typeof(bool) });
            if (linkTo == null || setAlign == null) return false;

            try
            {
                linkTo.Invoke(fal, new object[] { humanoidBone, "" });
                setAlign.Invoke(fal, new object[] { true });
            }
            catch (Exception e)
            {
                Debug.LogWarning("VRCFury Armature Link config threw: " + e.Message);
                return false;
            }

            Debug.Log("Attached VRCFury Armature Link on " + anchor.name + " -> " + boneName + " (aligned).");
            return true;
        }

        // --- VRCFury Full Controller wiring (reflection, optional) -------------

        // Uses VRCFury's public API (com.vrcfury.api.FuryComponents) to add a
        // Full Controller component on the rig root with the generated params
        // asset already plugged into the Parameters list. Without this, the
        // VRCRaycast components publish values to the animator but never to OSC.
        private static bool TryAddVRCFuryFullController(GameObject host, ScriptableObject paramsAsset)
        {
            var furyComponents = FindType("com.vrcfury.api.FuryComponents");
            if (furyComponents == null) return false;

            var create = furyComponents.GetMethod(
                "CreateFullController",
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(GameObject) },
                null
            );
            if (create == null) return false;

            object fc;
            try { fc = create.Invoke(null, new object[] { host }); }
            catch (Exception e) { Debug.LogWarning("VRCFury CreateFullController threw: " + e.Message); return false; }
            if (fc == null) return false;

            var addParams = fc.GetType().GetMethod("AddParams", BindingFlags.Public | BindingFlags.Instance);
            if (addParams == null) return false;

            try { addParams.Invoke(fc, new object[] { paramsAsset }); }
            catch (Exception e) { Debug.LogWarning("VRCFury AddParams threw: " + e.Message); return false; }

            Debug.Log("Attached VRCFury Full Controller with GabrielSensorParameters.");
            return true;
        }

        // --- Expression Parameters asset ---------------------------------------

        private static ScriptableObject BuildExpressionParameters()
        {
            var t = FindType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionParameters");
            if (t == null)
            {
                Debug.LogWarning("VRCExpressionParameters type not found, skipping params asset.");
                return null;
            }

            var asset = ScriptableObject.CreateInstance(t);

            // build the parameter list as a managed array. Field name in SDK3
            // is `parameters`. Inner element type is VRCExpressionParameters.Parameter.
            var paramsField = t.GetField("parameters", BindingFlags.Public | BindingFlags.Instance);
            if (paramsField == null) return asset;

            var elementType = paramsField.FieldType.GetElementType();
            if (elementType == null) return asset;

            var entries = new List<object>();
            foreach (var ray in RAYS)
            {
                entries.Add(MakeParam(elementType, ray.Name + "_Hit", valueType: 2, synced: false));      // Bool=2
                entries.Add(MakeParam(elementType, ray.Name + "_Distance", valueType: 1, synced: false)); // Float=1
                entries.Add(MakeParam(elementType, ray.Name + "_Ratio", valueType: 1, synced: false));
            }

            var arr = Array.CreateInstance(elementType, entries.Count);
            for (int i = 0; i < entries.Count; i++) arr.SetValue(entries[i], i);
            paramsField.SetValue(asset, arr);

            return asset;
        }

        private static object MakeParam(Type elementType, string name, int valueType, bool synced)
        {
            var p = Activator.CreateInstance(elementType);
            SetField(p, "name", name);
            // valueType enum: Int=0, Float=1, Bool=2 in current SDK3
            TrySetEnum(p, new[] { "valueType" }, null, fallbackInts: new[] { valueType });
            SetField(p, "saved", false);
            SetField(p, "defaultValue", 0f);
            // newer SDK has networkSynced (bool) instead of synced
            if (!SetField(p, "networkSynced", synced))
                SetField(p, "synced", synced);
            return p;
        }

        // --- reflection helpers ------------------------------------------------

        private static Type FindType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(fullName, false);
                if (t != null) return t;
            }
            return null;
        }

        private static bool SetField(object target, string name, object value)
        {
            var t = target.GetType();
            var f = t.GetField(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
            if (f == null) return false;
            try { f.SetValue(target, value); return true; }
            catch { return false; }
        }

        private static bool TrySetFloat(object target, string[] candidateNames, float value)
        {
            foreach (var name in candidateNames)
                if (SetField(target, name, value)) return true;
            return false;
        }

        private static bool TrySetEnum(object target, string[] candidateNames, string enumName, int[] fallbackInts = null)
        {
            var t = target.GetType();
            foreach (var name in candidateNames)
            {
                var f = t.GetField(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.NonPublic);
                if (f == null) continue;
                try
                {
                    if (enumName != null && f.FieldType.IsEnum && Enum.IsDefined(f.FieldType, enumName))
                    {
                        f.SetValue(target, Enum.Parse(f.FieldType, enumName));
                        return true;
                    }
                    if (fallbackInts != null && f.FieldType.IsEnum)
                    {
                        f.SetValue(target, Enum.ToObject(f.FieldType, fallbackInts[0]));
                        return true;
                    }
                    if (fallbackInts != null)
                    {
                        f.SetValue(target, fallbackInts[0]);
                        return true;
                    }
                }
                catch { /* try next candidate */ }
            }
            return false;
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            var parts = path.Split('/');
            var current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                var next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }
    }
}
#endif
