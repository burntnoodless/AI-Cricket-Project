import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) at point b, formed by lines ab and bc."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_vector_angle(point1, point2):
    """Calculates the angle of a vector formed by two points, relative to the horizontal axis."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

def detect_shot_type(phase_data):
    """
    Detect the type of cricket shot based on movement patterns.
    Returns: shot_type, confidence_score
    """
    try:
        # Calculate key indicators
        body_rotation = 0
        if phase_data['STANCE']['body_directions'] and phase_data['DOWNSWING']['body_directions']:
            body_rotation = abs(np.mean(phase_data['STANCE']['body_directions']) -
                              phase_data['DOWNSWING']['body_directions'][-1])

        weight_shift = 0
        if phase_data['STANCE']['weight_transfers'] and phase_data['DOWNSWING']['weight_transfers']:
            weight_shift = (phase_data['DOWNSWING']['weight_transfers'][-1] -
                          np.mean(phase_data['STANCE']['weight_transfers'])) * 100

        # Get backlift height indicator (wrist position relative to shoulder)
        backlift_height = len(phase_data['BACKLIFT']['knee_angles']) / max(len(phase_data['STANCE']['knee_angles']), 1)

        # Shot classification logic
        shot_type = "UNKNOWN"
        confidence = 0.0

        # DEFENSE: Minimal body rotation, minimal weight transfer, low backlift
        if body_rotation < 15 and abs(weight_shift) < 5 and backlift_height < 0.5:
            shot_type = "DEFENSIVE"
            confidence = 0.85

        # DRIVE: Forward weight transfer, moderate body rotation, full extension
        elif weight_shift > 5 and 15 <= body_rotation <= 45:
            shot_type = "DRIVE"
            confidence = 0.80

        # PULL/HOOK: High body rotation, weight transfer backwards or neutral
        elif body_rotation > 50 and weight_shift < 5:
            shot_type = "PULL/HOOK"
            confidence = 0.75

        # CUT: High body rotation, minimal forward movement, quick downswing
        elif body_rotation > 45 and weight_shift < 3 and len(phase_data['DOWNSWING']['knee_angles']) < len(phase_data['BACKLIFT']['knee_angles']):
            shot_type = "CUT"
            confidence = 0.75

        # SWEEP: Lower body position, significant rotation
        elif body_rotation > 60:
            shot_type = "SWEEP"
            confidence = 0.70

        # LOFTED SHOT: High backlift, significant body rotation
        elif backlift_height > 1.2 and body_rotation > 30:
            shot_type = "LOFTED"
            confidence = 0.70

        # Default to forward shot if unclear
        else:
            shot_type = "FORWARD SHOT"
            confidence = 0.60

        return shot_type, confidence

    except Exception as e:
        return "UNKNOWN", 0.0

def analyze_cricket_shot(video_path):
    """Analyzes a cricket shot video to provide professional-level coaching feedback."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    stage = "UNKNOWN"
    follow_through_frames = 0  # Counter for frames after shot completion
    FOLLOW_THROUGH_BUFFER = 30  # Number of frames to capture after follow-through detected

    # Data structure to hold metrics for each phase
    phase_data = {
        'STANCE': {'knee_angles': [], 'leg_directions': [], 'head_directions': [], 'body_directions': [], 'weight_transfers': []},
        'BACKLIFT': {'knee_angles': [], 'leg_directions': [], 'head_directions': [], 'body_directions': [], 'weight_transfers': []},
        'DOWNSWING': {'knee_angles': [], 'leg_directions': [], 'head_directions': [], 'body_directions': [], 'weight_transfers': [], 'elbow_angles': []},
        'FOLLOW_THROUGH': {'knee_angles': [], 'leg_directions': [], 'head_directions': [], 'body_directions': [], 'weight_transfers': []}
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # --- Get all required landmark coordinates ---
                # Arms & Shoulders
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                # Head
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                # Hips, Knees, Ankles
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # --- Shot Phase Detection Logic ---
                if stage == "UNKNOWN": stage = "STANCE"
                if stage in ["STANCE", "BACKLIFT"] and left_wrist[1] < nose[1] * 1.1: stage = "BACKLIFT"
                if stage == "BACKLIFT" and left_wrist[1] > nose[1] * 0.9: stage = "DOWNSWING"
                if stage == "DOWNSWING" and left_wrist[0] > right_shoulder[0]: stage = "FOLLOW_THROUGH" # Wrist crosses body

                # --- Calculate advanced metrics for the current frame ---
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                front_leg_direction = calculate_vector_angle(left_knee, left_hip) # Hip relative to knee
                body_direction = calculate_vector_angle(right_shoulder, left_shoulder)
                shoulder_midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                head_direction = calculate_vector_angle(shoulder_midpoint, nose)

                # Weight transfer calculation
                hip_midpoint_x = (left_hip[0] + right_hip[0]) / 2
                ankle_midpoint_x = (left_ankle[0] + right_ankle[0]) / 2
                weight_transfer = hip_midpoint_x - ankle_midpoint_x

                # --- Store metrics based on the current phase ---
                if stage in phase_data:
                    current_phase_data = phase_data[stage]
                    current_phase_data['knee_angles'].append(left_knee_angle)
                    current_phase_data['leg_directions'].append(front_leg_direction)
                    current_phase_data['body_directions'].append(body_direction)
                    current_phase_data['head_directions'].append(head_direction)
                    current_phase_data['weight_transfers'].append(weight_transfer)
                    if stage == 'DOWNSWING':
                        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        current_phase_data['elbow_angles'].append(elbow_angle)

                # --- Visualization ---
                cv2.rectangle(image, (0,0), (image.shape[1], 40), (245, 117, 16), -1)
                cv2.putText(image, f'PHASE: {stage}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Track follow-through frames and stop after buffer
                if stage == "FOLLOW_THROUGH":
                    follow_through_frames += 1

            except Exception:
                pass

            yield image, None

            # Stop analysis after capturing enough follow-through frames
            if stage == "FOLLOW_THROUGH" and follow_through_frames >= FOLLOW_THROUGH_BUFFER:
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- Final Metrics Processing ---
    final_metrics = {}
    try:
        def get_avg(data): return np.mean(data) if data else 0
        def get_last(data): return data[-1] if data else 0

        final_metrics['stance_body_direction'] = get_avg(phase_data['STANCE']['body_directions'])
        final_metrics['impact_body_direction'] = get_last(phase_data['DOWNSWING']['body_directions'])
        final_metrics['body_rotation_over_time'] = phase_data['STANCE']['body_directions'] + phase_data['BACKLIFT']['body_directions'] + phase_data['DOWNSWING']['body_directions'] + phase_data['FOLLOW_THROUGH']['body_directions']

        final_metrics['stance_leg_direction'] = get_avg(phase_data['STANCE']['leg_directions'])
        final_metrics['impact_leg_direction'] = get_last(phase_data['DOWNSWING']['leg_directions'])

        initial_weight = get_avg(phase_data['STANCE']['weight_transfers'])
        impact_weight = get_last(phase_data['DOWNSWING']['weight_transfers'])
        final_metrics['weight_transfer_amount'] = (impact_weight - initial_weight) * 100 # As a percentage for easier interpretation

        final_metrics['max_elbow_angle'] = max(phase_data['DOWNSWING']['elbow_angles']) if phase_data['DOWNSWING']['elbow_angles'] else 0
        final_metrics['stance_knee_angle'] = get_avg(phase_data['STANCE']['knee_angles'])
        final_metrics['downswing_knee_angle'] = get_avg(phase_data['DOWNSWING']['knee_angles'])
        final_metrics['stance_head_direction'] = get_avg(phase_data['STANCE']['head_directions'])
        final_metrics['downswing_head_direction'] = get_avg(phase_data['DOWNSWING']['head_directions'])

        # Detect shot type
        shot_type, confidence = detect_shot_type(phase_data)
        final_metrics['shot_type'] = shot_type
        final_metrics['shot_confidence'] = confidence

        # Additional advanced metrics
        final_metrics['body_rotation_total'] = abs(final_metrics['stance_body_direction'] - final_metrics['impact_body_direction'])
        final_metrics['head_movement'] = abs(final_metrics['stance_head_direction'] - final_metrics['downswing_head_direction'])
        final_metrics['knee_bracing'] = final_metrics['downswing_knee_angle'] - final_metrics['stance_knee_angle']

    except Exception as e:
        print(f"Could not calculate all final metrics: {e}")

    yield None, final_metrics
