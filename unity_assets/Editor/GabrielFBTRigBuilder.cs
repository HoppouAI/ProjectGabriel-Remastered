#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace ProjectGabriel.Editor
{
    /// <summary>
    /// Rebuilds the complete DesktopFBT puppet rig from nothing: every clip,
    /// the Action controller, the VRC expression parameters + menu, the hips
    /// rotation proxy chain and constraint, and the avatar descriptor wiring.
    /// This is the reference implementation of the rig the motion server
    /// streams to (motion_server/retarget.py mirrors these exact conventions).
    ///
    /// Menu: Tools > ProjectGabriel > Build Desktop FBT Rig   (avatar selected)
    ///        Tools > ProjectGabriel > Export muscle_ranges.json
    ///
    /// HOW IT WORKS (read this before touching anything):
    ///   - 29 synced float params (FBT/*) + 1 bool (FBT/Enable) = 233/256 bits.
    ///   - One Direct Blend Tree in the Action layer, every child weighted by
    ///     FBT/One (always 1). Each muscle param is a 1D tree lerping a _min
    ///     clip (param -1) to a _max clip (param +1). Muscle clips write
    ///     humanoid muscle curves, weighted down the chain (eg SpineFB writes
    ///     Spine 1.0 / Chest 0.7 / UpperChest 0.5).
    ///   - HipsY is RootT.y (humanoid body height). This survives the DBT.
    ///   - Hips rotation CANNOT use RootQ: unity nlerp-averages body rotation
    ///     across ALL blend tree children, so ~30 muscle clips each voting
    ///     "upright" crush 90 deg to ~3 deg. RootQ from layers above 0 is
    ///     discarded outright and muscle layers below stomp it. Instead two
    ///     proxy transforms under the avatar root get generic euler curves
    ///     (those blend per-curve, zero dilution) and a VRCRotationConstraint
    ///     on the Hips bone follows them. Its GlobalWeight is animated to 1
    ///     only inside the puppet state; write-defaults returns it to the
    ///     scene value 0 when the puppet disengages.
    ///   - Puppet state behaviours: tracking -> Animation for all body parts,
    ///     playable layer weight -> 1. Locomotion stays ENABLED so OSC move
    ///     inputs still drive the capsule while the puppet animates in place.
    /// </summary>
    public static class GabrielFBTRigBuilder
    {
        private const string OUT_DIR = "Assets/HoppouAI/DesktopFBT";
        private const string CLIP_DIR = OUT_DIR + "/Clips";

        // hips rotation range mapped over param -1..1. retarget.py
        // HIPS_PITCH_MAX_DEG / HIPS_ROLL_MAX_DEG must match.
        private const float PITCH_RANGE_DEG = 90f;
        private const float ROLL_RANGE_DEG = 90f;

        // HipsY drop below standing at param -1, in RootT units. derived as
        // smpl_meters * 1.05346 (in-game fudge measured against a 1.0148
        // humanScale avatar). retarget.py hipsy_down_m/up_m must be the smpl
        // values. down 1.00 puts the pelvis exactly on the floor at -1, up
        // 0.80 covers a real jump apex (dart lifts the pelvis ~0.74m).
        private const float HIPSY_DOWN_SMPL_M = 1.00f;
        private const float HIPSY_UP_SMPL_M = 0.80f;
        private const float HIPSY_DOWN_RIG = HIPSY_DOWN_SMPL_M * 1.05346f;
        private const float HIPSY_UP_RIG = HIPSY_UP_SMPL_M * 1.05346f;

        // param -> (unity muscle name, chain weight). mirrors PARAM_MUSCLES
        // in motion_server/retarget.py, keep the two in sync.
        private static readonly (string param, (string muscle, float w)[] chain)[] MUSCLE_PARAMS =
        {
            ("SpineFB", new[] { ("Spine Front-Back", 1f), ("Chest Front-Back", 0.7f), ("UpperChest Front-Back", 0.5f) }),
            ("SpineLR", new[] { ("Spine Left-Right", 1f), ("Chest Left-Right", 0.7f), ("UpperChest Left-Right", 0.5f) }),
            ("SpineTW", new[] { ("Spine Twist Left-Right", 1f), ("Chest Twist Left-Right", 0.7f), ("UpperChest Twist Left-Right", 0.5f) }),
            ("HeadNod", new[] { ("Neck Nod Down-Up", 0.6f), ("Head Nod Down-Up", 1f) }),
            ("HeadTilt", new[] { ("Neck Tilt Left-Right", 0.6f), ("Head Tilt Left-Right", 1f) }),
            ("HeadTurn", new[] { ("Neck Turn Left-Right", 0.6f), ("Head Turn Left-Right", 1f) }),
            ("LArmUp", new[] { ("Left Shoulder Down-Up", 0.5f), ("Left Arm Down-Up", 1f) }),
            ("LArmFB", new[] { ("Left Shoulder Front-Back", 0.5f), ("Left Arm Front-Back", 1f) }),
            ("LArmTW", new[] { ("Left Arm Twist In-Out", 1f) }),
            ("LElbow", new[] { ("Left Forearm Stretch", 1f) }),
            ("LWristUD", new[] { ("Left Hand Down-Up", 1f) }),
            ("LWristIO", new[] { ("Left Hand In-Out", 1f) }),
            ("RArmUp", new[] { ("Right Shoulder Down-Up", 0.5f), ("Right Arm Down-Up", 1f) }),
            ("RArmFB", new[] { ("Right Shoulder Front-Back", 0.5f), ("Right Arm Front-Back", 1f) }),
            ("RArmTW", new[] { ("Right Arm Twist In-Out", 1f) }),
            ("RElbow", new[] { ("Right Forearm Stretch", 1f) }),
            ("RWristUD", new[] { ("Right Hand Down-Up", 1f) }),
            ("RWristIO", new[] { ("Right Hand In-Out", 1f) }),
            ("LLegFB", new[] { ("Left Upper Leg Front-Back", 1f) }),
            ("LLegIO", new[] { ("Left Upper Leg In-Out", 1f) }),
            ("LKnee", new[] { ("Left Lower Leg Stretch", 1f) }),
            ("LFootUD", new[] { ("Left Foot Up-Down", 1f) }),
            ("RLegFB", new[] { ("Right Upper Leg Front-Back", 1f) }),
            ("RLegIO", new[] { ("Right Upper Leg In-Out", 1f) }),
            ("RKnee", new[] { ("Right Lower Leg Stretch", 1f) }),
            ("RFootUD", new[] { ("Right Foot Up-Down", 1f) }),
        };

        [MenuItem("Tools/ProjectGabriel/Build Desktop FBT Rig")]
        public static void Build()
        {
            var avatar = Selection.activeGameObject;
            var animator = avatar ? avatar.GetComponent<Animator>() : null;
            if (animator == null || !animator.isHuman)
            {
                EditorUtility.DisplayDialog("FBT Rig", "Select a humanoid avatar root first.", "ok");
                return;
            }

            Directory.CreateDirectory(CLIP_DIR);

            // measure standing body height so HipsY mid matches this avatar
            var handler = new HumanPoseHandler(animator.avatar, avatar.transform);
            var pose = new HumanPose();
            handler.GetHumanPose(ref pose);
            float midY = pose.bodyPosition.y;

            var hips = animator.GetBoneTransform(HumanBodyBones.Hips);
            string hipsPath = AnimationUtility.CalculateTransformPath(hips, avatar.transform);

            // ---- clips ----
            var clips = new Dictionary<string, AnimationClip>();
            foreach (var (param, chain) in MUSCLE_PARAMS)
            {
                clips[param + "_min"] = MuscleClip($"FBT_{param}_min", chain, -1f);
                clips[param + "_max"] = MuscleClip($"FBT_{param}_max", chain, +1f);
            }
            clips["HipsY_min"] = FloatClip("FBT_HipsY_min", "", typeof(Animator), "RootT.y", midY - HIPSY_DOWN_RIG);
            clips["HipsY_mid"] = FloatClip("FBT_HipsY_mid", "", typeof(Animator), "RootT.y", midY);
            clips["HipsY_max"] = FloatClip("FBT_HipsY_max", "", typeof(Animator), "RootT.y", midY + HIPSY_UP_RIG);
            // proxy rotation: pitch +1 = +X euler (lean forward), roll +1 = -Z (lean right)
            clips["Pitch_min"] = FloatClip("FBT_HipsPitch_min", "FBT_HipsPitchProxy", typeof(Transform), "localEulerAnglesRaw.x", -PITCH_RANGE_DEG);
            clips["Pitch_mid"] = FloatClip("FBT_HipsPitch_mid", "FBT_HipsPitchProxy", typeof(Transform), "localEulerAnglesRaw.x", 0f);
            clips["Pitch_max"] = FloatClip("FBT_HipsPitch_max", "FBT_HipsPitchProxy", typeof(Transform), "localEulerAnglesRaw.x", +PITCH_RANGE_DEG);
            clips["Roll_min"] = FloatClip("FBT_HipsRoll_min", "FBT_HipsPitchProxy/FBT_HipsRollProxy", typeof(Transform), "localEulerAnglesRaw.z", +ROLL_RANGE_DEG);
            clips["Roll_mid"] = FloatClip("FBT_HipsRoll_mid", "FBT_HipsPitchProxy/FBT_HipsRollProxy", typeof(Transform), "localEulerAnglesRaw.z", 0f);
            clips["Roll_max"] = FloatClip("FBT_HipsRoll_max", "FBT_HipsPitchProxy/FBT_HipsRollProxy", typeof(Transform), "localEulerAnglesRaw.z", -ROLL_RANGE_DEG);
            clips["Empty"] = FloatClip("FBT_Empty", "", typeof(Animator), "DummyUnused", 0f);

            var conType = FindType("VRC.SDK3.Dynamics.Constraint.Components.VRCRotationConstraint");
            if (conType != null)
                clips["ConstraintOn"] = FloatClip("FBT_HipsRotConstraintOn", hipsPath, conType, "GlobalWeight", 1f);
            else
                Debug.LogWarning("[FBT] VRC constraint type missing, hips rotation will not work (no SDK?)");

            // ---- controller ----
            string ctrlPath = OUT_DIR + "/FBT_Action.controller";
            AssetDatabase.DeleteAsset(ctrlPath);
            var ctrl = AnimatorController.CreateAnimatorControllerAtPath(ctrlPath);
            ctrl.AddParameter("FBT/Enable", AnimatorControllerParameterType.Bool);
            var one = new AnimatorControllerParameter { name = "FBT/One", type = AnimatorControllerParameterType.Float, defaultFloat = 1f };
            ctrl.AddParameter(one);
            foreach (var (param, _) in MUSCLE_PARAMS)
                ctrl.AddParameter("FBT/" + param, AnimatorControllerParameterType.Float);
            foreach (var p in new[] { "FBT/HipsY", "FBT/HipsPitch", "FBT/HipsRoll" })
                ctrl.AddParameter(p, AnimatorControllerParameterType.Float);

            var sm = ctrl.layers[0].stateMachine;
            var idle = sm.AddState("Idle", new Vector3(250, 0, 0));
            idle.motion = clips["Empty"];
            var puppet = sm.AddState("Puppet", new Vector3(250, 120, 0));
            var reset = sm.AddState("Reset", new Vector3(250, 240, 0));
            reset.motion = clips["Empty"];
            sm.defaultState = idle;

            var master = new BlendTree { name = "FBT Root DBT", blendType = BlendTreeType.Direct, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(master, ctrl);
            puppet.motion = master;

            var children = new List<ChildMotion>();
            foreach (var (param, _) in MUSCLE_PARAMS)
                children.Add(DirectChild(Tree1D(ctrl, "FBT " + param, "FBT/" + param,
                    (clips[param + "_min"], -1f), (clips[param + "_max"], 1f))));
            children.Add(DirectChild(Tree1D(ctrl, "FBT HipsY", "FBT/HipsY",
                (clips["HipsY_min"], -1f), (clips["HipsY_mid"], 0f), (clips["HipsY_max"], 1f))));
            children.Add(DirectChild(Tree1D(ctrl, "FBT HipsPitch", "FBT/HipsPitch",
                (clips["Pitch_min"], -1f), (clips["Pitch_mid"], 0f), (clips["Pitch_max"], 1f))));
            children.Add(DirectChild(Tree1D(ctrl, "FBT HipsRoll2", "FBT/HipsRoll",
                (clips["Roll_min"], -1f), (clips["Roll_mid"], 0f), (clips["Roll_max"], 1f))));
            if (clips.ContainsKey("ConstraintOn"))
                children.Add(DirectChild(clips["ConstraintOn"]));
            master.children = children.ToArray();

            var toPuppet = idle.AddTransition(puppet);
            toPuppet.hasExitTime = false; toPuppet.duration = 0.2f; toPuppet.hasFixedDuration = true;
            toPuppet.AddCondition(AnimatorConditionMode.If, 0, "FBT/Enable");
            var toReset = puppet.AddTransition(reset);
            toReset.hasExitTime = false; toReset.duration = 0.2f; toReset.hasFixedDuration = true;
            toReset.AddCondition(AnimatorConditionMode.IfNot, 0, "FBT/Enable");
            var toIdle = reset.AddTransition(idle);
            toIdle.hasExitTime = true; toIdle.exitTime = 1f; toIdle.duration = 0f; toIdle.hasFixedDuration = true;

            // vrc state behaviours: puppet takes over tracking, reset gives it back
            AddBehaviour(puppet, "VRC.SDK3.Avatars.Components.VRCPlayableLayerControl",
                ("layer", 0), ("goalWeight", 1f), ("blendDuration", 0.25f));
            AddBehaviour(puppet, "VRC.SDK3.Avatars.Components.VRCAnimatorTrackingControl",
                ("trackingHead", 2), ("trackingLeftHand", 2), ("trackingRightHand", 2),
                ("trackingHip", 2), ("trackingLeftFoot", 2), ("trackingRightFoot", 2));
            AddBehaviour(puppet, "VRC.SDK3.Avatars.Components.VRCAnimatorLocomotionControl",
                ("disableLocomotion", false));
            AddBehaviour(reset, "VRC.SDK3.Avatars.Components.VRCPlayableLayerControl",
                ("layer", 0), ("goalWeight", 0f), ("blendDuration", 0.25f));
            AddBehaviour(reset, "VRC.SDK3.Avatars.Components.VRCAnimatorTrackingControl",
                ("trackingHead", 1), ("trackingLeftHand", 1), ("trackingRightHand", 1),
                ("trackingHip", 1), ("trackingLeftFoot", 1), ("trackingRightFoot", 1));
            AddBehaviour(reset, "VRC.SDK3.Avatars.Components.VRCAnimatorLocomotionControl",
                ("disableLocomotion", false));

            // ---- vrc params + menu ----
            var paramsAsset = BuildExpressionParameters(OUT_DIR + "/FBT_Params.asset");
            var menuAsset = BuildExpressionsMenu(OUT_DIR + "/FBT_Menu.asset");

            // ---- avatar hierarchy: proxies + constraint ----
            var oldProxy = avatar.transform.Find("FBT_HipsPitchProxy");
            if (oldProxy != null) UnityEngine.Object.DestroyImmediate(oldProxy.gameObject);
            var pitchGo = new GameObject("FBT_HipsPitchProxy");
            pitchGo.transform.SetParent(avatar.transform, false);
            pitchGo.transform.localPosition = avatar.transform.InverseTransformPoint(hips.position);
            var rollGo = new GameObject("FBT_HipsRollProxy");
            rollGo.transform.SetParent(pitchGo.transform, false);

            if (conType != null)
            {
                var oldCon = hips.GetComponent(conType);
                if (oldCon != null) UnityEngine.Object.DestroyImmediate(oldCon);
                var con = hips.gameObject.AddComponent(conType);
                var srcType = FindType("VRC.Dynamics.VRCConstraintSource");
                var sourcesField = conType.GetField("Sources");
                object sources = sourcesField.GetValue(con);
                object src = Activator.CreateInstance(srcType);
                srcType.GetField("SourceTransform").SetValue(src, rollGo.transform);
                srcType.GetField("Weight").SetValue(src, 1f);
                sources.GetType().GetMethod("Add", new[] { srcType }).Invoke(sources, new[] { src });
                sourcesField.SetValue(con, sources);  // boxed struct list, must write back
                var offQ = Quaternion.Inverse(rollGo.transform.rotation) * hips.rotation;
                conType.GetField("RotationOffset").SetValue(con, offQ.eulerAngles);
                conType.GetField("RotationAtRest").SetValue(con, hips.localRotation.eulerAngles);
                conType.GetField("IsActive").SetValue(con, true);
                conType.GetField("GlobalWeight").SetValue(con, 0f);  // released until puppet engages
                conType.GetField("Locked").SetValue(con, true);
                EditorUtility.SetDirty(con);
            }

            // ---- descriptor wiring ----
            WireDescriptor(avatar, ctrl, paramsAsset, menuAsset);

            AssetDatabase.SaveAssets();
            Debug.Log($"[FBT] rig built for '{avatar.name}': midY={midY:F5}, hips='{hipsPath}', {children.Count} tree children. Reupload the avatar.");
        }

        [MenuItem("Tools/ProjectGabriel/Export muscle_ranges.json")]
        public static void ExportRanges()
        {
            var avatar = Selection.activeGameObject;
            var animator = avatar ? avatar.GetComponent<Animator>() : null;
            if (animator == null || !animator.isHuman)
            {
                EditorUtility.DisplayDialog("FBT Rig", "Select a humanoid avatar root first.", "ok");
                return;
            }
            var handler = new HumanPoseHandler(animator.avatar, avatar.transform);
            var pose = new HumanPose();
            handler.GetHumanPose(ref pose);

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine($"  \"humanScale\": {animator.humanScale:R},");
            sb.AppendLine($"  \"hipsYUpMeters\": {HIPSY_UP_SMPL_M:R},");
            sb.AppendLine($"  \"hipsYDownMeters\": {HIPSY_DOWN_SMPL_M:R},");
            sb.AppendLine("  \"muscles\": {");
            var names = HumanTrait.MuscleName;
            for (int i = 0; i < names.Length; i++)
            {
                string comma = i < names.Length - 1 ? "," : "";
                sb.AppendLine($"    \"{names[i]}\": {{\"min\": {HumanTrait.GetMuscleDefaultMin(i):R}, \"max\": {HumanTrait.GetMuscleDefaultMax(i):R}}}{comma}");
            }
            sb.AppendLine("  }");
            sb.AppendLine("}");
            string path = EditorUtility.SaveFilePanel("Export muscle ranges", "", "muscle_ranges.json", "json");
            if (!string.IsNullOrEmpty(path))
            {
                File.WriteAllText(path, sb.ToString());
                Debug.Log($"[FBT] wrote {path} (drop into motion_server/)");
            }
        }

        // ---------- helpers ----------

        private static AnimationClip MuscleClip(string name, (string muscle, float w)[] chain, float sign)
        {
            var clip = NewClip(name);
            foreach (var (muscle, w) in chain)
                clip.SetCurve("", typeof(Animator), muscle, Flat(sign * w));
            SaveClip(clip);
            return clip;
        }

        private static AnimationClip FloatClip(string name, string path, Type type, string prop, float v)
        {
            var clip = NewClip(name);
            var binding = new EditorCurveBinding { path = path, type = type, propertyName = prop };
            AnimationUtility.SetEditorCurve(clip, binding, Flat(v));
            SaveClip(clip);
            return clip;
        }

        private static AnimationClip NewClip(string name)
        {
            string p = $"{CLIP_DIR}/{name}.anim";
            AssetDatabase.DeleteAsset(p);
            var clip = new AnimationClip { name = name };
            return clip;
        }

        private static void SaveClip(AnimationClip clip)
        {
            AssetDatabase.CreateAsset(clip, $"{CLIP_DIR}/{clip.name}.anim");
        }

        private static AnimationCurve Flat(float v) =>
            new AnimationCurve(new Keyframe(0f, v), new Keyframe(1f / 60f, v));

        private static BlendTree Tree1D(AnimatorController owner, string name, string param, params (AnimationClip clip, float th)[] kids)
        {
            var t = new BlendTree { name = name, blendType = BlendTreeType.Simple1D, blendParameter = param, useAutomaticThresholds = false, hideFlags = HideFlags.HideInHierarchy };
            foreach (var (clip, th) in kids) t.AddChild(clip, th);
            AssetDatabase.AddObjectToAsset(t, owner);
            return t;
        }

        private static ChildMotion DirectChild(Motion m) =>
            new ChildMotion { motion = m, directBlendParameter = "FBT/One", timeScale = 1f };

        private static void AddBehaviour(AnimatorState state, string typeName, params (string field, object value)[] fields)
        {
            var t = FindType(typeName);
            if (t == null) { Debug.LogWarning($"[FBT] behaviour type missing: {typeName}"); return; }
            var b = state.AddStateMachineBehaviour(t);
            foreach (var (field, value) in fields)
            {
                var f = t.GetField(field);
                if (f == null) continue;
                object v = f.FieldType.IsEnum ? Enum.ToObject(f.FieldType, Convert.ToInt32(value)) : Convert.ChangeType(value, f.FieldType);
                f.SetValue(b, v);
            }
        }

        private static UnityEngine.Object BuildExpressionParameters(string path)
        {
            var pType = FindType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionParameters");
            if (pType == null) { Debug.LogWarning("[FBT] VRCExpressionParameters type missing"); return null; }
            var entryType = pType.GetNestedType("Parameter");
            var asset = ScriptableObject.CreateInstance(pType);

            var allNames = new List<string> { "FBT/Enable" };
            foreach (var (param, _) in MUSCLE_PARAMS) allNames.Add("FBT/" + param);
            allNames.AddRange(new[] { "FBT/HipsY", "FBT/HipsPitch", "FBT/HipsRoll" });

            var arr = Array.CreateInstance(entryType, allNames.Count);
            for (int i = 0; i < allNames.Count; i++)
            {
                object e = Activator.CreateInstance(entryType);
                entryType.GetField("name").SetValue(e, allNames[i]);
                // valueType enum: Int=0 Float=1 Bool=2
                var vtField = entryType.GetField("valueType");
                vtField.SetValue(e, Enum.ToObject(vtField.FieldType, i == 0 ? 2 : 1));
                entryType.GetField("saved").SetValue(e, false);
                entryType.GetField("defaultValue").SetValue(e, 0f);
                var ns = entryType.GetField("networkSynced");
                if (ns != null) ns.SetValue(e, true);
                arr.SetValue(e, i);
            }
            pType.GetField("parameters").SetValue(asset, arr);
            AssetDatabase.DeleteAsset(path);
            AssetDatabase.CreateAsset(asset, path);
            return asset;
        }

        private static UnityEngine.Object BuildExpressionsMenu(string path)
        {
            var mType = FindType("VRC.SDK3.Avatars.ScriptableObjects.VRCExpressionsMenu");
            if (mType == null) { Debug.LogWarning("[FBT] VRCExpressionsMenu type missing"); return null; }
            var cType = mType.GetNestedType("Control");
            var asset = ScriptableObject.CreateInstance(mType);

            object control = Activator.CreateInstance(cType);
            cType.GetField("name").SetValue(control, "FBT Puppet");
            var tField = cType.GetField("type");
            tField.SetValue(control, Enum.ToObject(tField.FieldType, 102)); // Toggle
            var prmType = cType.GetNestedType("Parameter");
            object prm = Activator.CreateInstance(prmType);
            prmType.GetField("name").SetValue(prm, "FBT/Enable");
            cType.GetField("parameter").SetValue(control, prm);
            cType.GetField("value").SetValue(control, 1f);

            var controlsField = mType.GetField("controls");
            var list = (IList)controlsField.GetValue(asset);
            list.Add(control);

            AssetDatabase.DeleteAsset(path);
            AssetDatabase.CreateAsset(asset, path);
            return asset;
        }

        private static void WireDescriptor(GameObject avatar, AnimatorController ctrl, UnityEngine.Object prms, UnityEngine.Object menu)
        {
            var dType = FindType("VRC.SDK3.Avatars.Components.VRCAvatarDescriptor");
            var desc = dType != null ? avatar.GetComponent(dType) : null;
            if (desc == null) { Debug.LogWarning("[FBT] no VRCAvatarDescriptor, wire the Action layer + params + menu manually"); return; }

            var so = new SerializedObject(desc);
            so.FindProperty("customizeAnimationLayers").boolValue = true;
            var layers = so.FindProperty("baseAnimationLayers");
            for (int i = 0; i < layers.arraySize; i++)
            {
                var e = layers.GetArrayElementAtIndex(i);
                if (e.FindPropertyRelative("type").intValue == 4) // Action
                {
                    e.FindPropertyRelative("animatorController").objectReferenceValue = ctrl;
                    e.FindPropertyRelative("isDefault").boolValue = false;
                }
            }
            so.FindProperty("customExpressions").boolValue = true;
            if (prms != null) so.FindProperty("expressionParameters").objectReferenceValue = prms;
            if (menu != null) so.FindProperty("expressionsMenu").objectReferenceValue = menu;
            so.ApplyModifiedProperties();
        }

        private static Type FindType(string fullName)
        {
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(fullName);
                if (t != null) return t;
            }
            return null;
        }
    }
}
#endif
