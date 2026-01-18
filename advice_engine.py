"""
AI Cricket Coach - Advanced Advice Generation Engine
Converts raw metrics into actionable, shot-specific coaching feedback
"""

from typing import Dict, List, Tuple
import random
import os

# OpenAI API Configuration (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = None

def get_openai_client():
    """Initialize OpenAI client with API key (if available)"""
    global openai_client
    if openai_client is None and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            pass
    return openai_client

class CricketAdviceEngine:
    """Generates personalized, context-aware coaching advice based on cricket shot analysis metrics"""

    def __init__(self):
        # Cricket-specific optimal thresholds based on coaching best practices
        self.thresholds = {
            'elbow_angle': {'optimal_min': 160, 'optimal_max': 180, 'poor_max': 140},
            'weight_transfer': {'optimal_min': 5, 'optimal_max': 20, 'poor_max': 0},
            'body_rotation': {'optimal_min': 20, 'optimal_max': 60, 'poor_min': 5, 'poor_max': 90},
            'head_stability': {'optimal_max': 10.0, 'poor_min': 25.0},
            'knee_angle_stance': {'optimal_min': 120, 'optimal_max': 150, 'poor_min': 100, 'poor_max': 170},
            'knee_angle_downswing': {'optimal_min': 160, 'optimal_max': 180, 'poor_max': 140},
            'knee_bracing': {'optimal_min': 10, 'optimal_max': 30}
        }

        # Shot-specific coaching advice - comprehensive database
        self.shot_specific_advice = {
            'DRIVE': {
                'description': 'The drive is the cornerstone of classical batting technique. A flowing, elegant stroke played to full-length deliveries, it requires perfect synchronization of footwork, head position, and bat swing. The front-foot drive is considered the measure of a batsman\'s technical proficiency.',
                'key_focus': [
                    'Full front arm extension through the shot',
                    'Balanced weight transfer onto front foot',
                    'Head still and over the ball at impact',
                    'High elbow position maintaining bat control',
                    'Front foot pointing towards the ball',
                    'Smooth follow-through towards target',
                    'Eyes level throughout the stroke',
                    'Front knee bent but firm at impact'
                ],
                'common_flaws': {
                    'low_elbow': 'For drives, maintaining a high elbow and straight front arm is crucial for timing and power. A dropped elbow causes the bat face to close',
                    'insufficient_weight': 'Get more weight transfer forward into the shot - you should feel your momentum moving towards the ball. Hanging back reduces power significantly',
                    'head_movement': 'Keep your head still and over the ball throughout the drive - head movement disrupts timing and causes edges',
                    'lunging': 'Avoid over-stretching. Your head should lead the movement, not your front foot. Lunging creates balance issues',
                    'closed_face': 'Ensure bat face stays open through impact. Check your grip pressure and top hand position',
                    'falling_away': 'Keep your back foot grounded until after impact. Falling away opens gaps in your technique'
                },
                'drills': [
                    'Front foot drives off a bowling machine at 70% pace - focus on head position',
                    'Shadow batting with mirror - check elbow height at impact point',
                    'Throw-downs with tennis ball - exaggerate weight transfer',
                    'Drive against wall with taped target - practice follow-through direction'
                ]
            },
            'PULL/HOOK': {
                'description': 'The pull and hook are aggressive back-foot shots played to short-pitched deliveries. The pull is played to balls at chest height with a horizontal bat, while the hook is played to balls at head height. Both require excellent reflexes, quick weight transfer, and precise timing to control the ball\'s trajectory.',
                'key_focus': [
                    'Quick weight transfer to back foot',
                    'Strong hip and shoulder rotation',
                    'Head still and eyes on the ball',
                    'Balanced finish position',
                    'Back foot pivoting for power generation',
                    'Arms away from body for free swing',
                    'Roll wrists to keep ball down',
                    'Watch ball onto bat until contact'
                ],
                'common_flaws': {
                    'forward_weight': 'Pull shots require weight on the back foot - avoid leaning forward which reduces reaction time',
                    'insufficient_rotation': 'Generate more power through hip and shoulder rotation - your core drives this shot',
                    'head_falling': 'Keep your eyes on the ball and head upright - don\'t fall away from the shot',
                    'top_edging': 'Rolling wrists too early or too late causes top edges. Practice timing the wrist roll',
                    'stuck_on_crease': 'Get into position early. Late movement causes rushed, uncontrolled shots',
                    'stiff_arms': 'Keep arms relaxed for a free flowing swing. Tension kills bat speed'
                },
                'drills': [
                    'Short ball machine work - start at reduced pace and gradually increase',
                    'Tennis ball throws at head height - practice ducking vs playing',
                    'Pull shot shadow batting focusing on hip rotation',
                    'Reaction drills with colored balls - decide pull vs duck'
                ]
            },
            'CUT': {
                'description': 'The cut shot is a controlled horizontal bat stroke played to short, wide deliveries outside off stump. It requires excellent judgment of length and width, late bat movement, and precise wrist control. The square cut and late cut are variations based on timing and placement.',
                'key_focus': [
                    'Late bat movement for deception',
                    'Weight centered or slightly back',
                    'Strong wrist control at impact',
                    'Upper body rotation for power',
                    'Back foot moving across to the ball',
                    'High hands through the shot',
                    'Watch ball right onto the bat',
                    'Controlled follow-through'
                ],
                'common_flaws': {
                    'early_shot': 'Wait for the ball - cut shots require late execution. Early shots cause edges and mistimed hits',
                    'forward_weight': 'Keep weight slightly back and centered - forward movement restricts your ability to adjust',
                    'low_body_rotation': 'Use your shoulders to generate power and control - rotation creates the cut angle',
                    'fishing': 'Only cut balls you can hit on top of. Avoid reaching for balls too wide',
                    'hard_hands': 'Softer hands allow better placement. Death grip reduces control',
                    'bottom_hand_dominant': 'Top hand guides, bottom hand powers. Over-use of bottom hand causes cross-bat errors'
                },
                'drills': [
                    'Side-on throw-downs outside off stump - practice leaving vs cutting decision',
                    'Cut shot against spin bowling - develops timing and placement',
                    'Back foot movement drills - quick lateral steps',
                    'Wrist strengthening exercises for better bat control'
                ]
            },
            'DEFENSIVE': {
                'description': 'The defensive shot is the foundation of all batting. It\'s about survival, occupying the crease, and negating good deliveries. A solid defense frustrates bowlers and creates opportunities for scoring. The forward and back defensive are essential for building innings.',
                'key_focus': [
                    'Soft, relaxed hands to deaden the ball',
                    'Head directly over the ball at contact',
                    'Minimal backlift for control',
                    'Solid, balanced base',
                    'Bat angled down to cover bounce',
                    'Eyes level and watching the ball',
                    'Front elbow high and leading',
                    'Pad and bat close together'
                ],
                'common_flaws': {
                    'hard_hands': 'Relax your grip - defensive shots need soft hands to deaden the ball and prevent catches',
                    'high_backlift': 'Keep your backlift minimal for better control on defensive shots - high backlift creates timing issues',
                    'head_movement': 'Absolute head stillness is critical for defensive technique - any movement causes edges',
                    'bat_pad_gap': 'Keep bat and pad together - gaps between them invite LBW and bowled dismissals',
                    'pushing_at_ball': 'Let the ball come to you. Pushing creates edges to slip',
                    'weight_back': 'Get forward to smother spin and seam movement. Hanging back is dangerous'
                },
                'drills': [
                    'Defense against spin on turning pitches - practice smothering turn',
                    'Ball drop exercises - soft hands catching drill',
                    'Forward defense with eyes closed (after release) - trust technique',
                    'Defense against seam movement - focus on playing late'
                ]
            },
            'SWEEP': {
                'description': 'The sweep is a premeditated shot played against spin bowling, particularly to full-length deliveries. It involves getting low and using a horizontal bat to sweep the ball to the leg side. Variations include the paddle sweep and reverse sweep for different placements.',
                'key_focus': [
                    'Low body position with bent knees',
                    'Weight committed forward',
                    'Top hand controlling bat face',
                    'Head over front knee',
                    'Front pad outside line of ball',
                    'Roll bat face for placement',
                    'Committed decisive movement',
                    'Use pad as secondary defense'
                ],
                'common_flaws': {
                    'high_body': 'Get lower to the ball - sweep shots require a low body position to get under the bounce',
                    'poor_balance': 'Maintain balance throughout - don\'t overbalance or fall over which exposes you to LBW',
                    'bat_control': 'Control the shot with your top hand for better placement - bottom hand dominance causes mishits',
                    'wrong_length': 'Only sweep full or good length balls. Sweeping short balls causes top edges',
                    'head_falling': 'Keep head still and over front knee. Falling head causes loss of ball tracking',
                    'tentative': 'Commit fully to the sweep. Half-hearted sweeps are dangerous'
                },
                'drills': [
                    'Sweep against underarm spin throws - build confidence',
                    'Knee strengthening for low position maintenance',
                    'Placement practice - cones at fine leg, square leg, backward square',
                    'Switch between sweep and pad-first defense based on line'
                ]
            },
            'LOFTED': {
                'description': 'Lofted shots are aggressive aerial strokes designed to clear the infield or hit boundaries. They require confident technique, excellent timing, and commitment. The straight drive over the bowler and the lofted on-drive are high-risk, high-reward shots that can change a game.',
                'key_focus': [
                    'Full, free swing through the ball',
                    'Strong, stable base throughout',
                    'Head still at point of contact',
                    'Complete follow-through high',
                    'Trust your technique and timing',
                    'Get to the pitch of the ball',
                    'Smooth acceleration through impact',
                    'Maintain balance on landing'
                ],
                'common_flaws': {
                    'timing': 'Focus on timing rather than power - let the bat swing do the work. Hitting too hard disrupts technique',
                    'balance': 'Maintain a strong, balanced base throughout the shot - poor balance causes mishits',
                    'head_position': 'Keep your head still at the point of contact - head movement changes the swing plane',
                    'slogging': 'Even lofted shots need technique. Wild swings reduce consistency dramatically',
                    'short_ball': 'Only loft full deliveries. Lofting short balls is high risk for top edges',
                    'not_getting_to_pitch': 'Use your feet to get to the pitch. Playing from the crease limits power and control'
                },
                'drills': [
                    'Lofted drives with tennis ball first - build confidence',
                    'Target practice - hit cones at different distances',
                    'Footwork drills - quick steps to pitch of ball',
                    'Balance exercises - single leg stability'
                ]
            },
            'FORWARD SHOT': {
                'description': 'A general forward playing shot that encompasses various front-foot strokes. The key is transferring weight onto the front foot while maintaining balance and keeping the head over the ball. This forms the basis for all attacking and defensive front-foot play.',
                'key_focus': [
                    'Front foot moving towards pitch of ball',
                    'Head leading the movement',
                    'Straight bat face at impact',
                    'Balanced weight transfer',
                    'Eyes level and watching ball',
                    'Front knee bent for stability',
                    'Back foot stays grounded initially',
                    'Smooth weight shift not lunge'
                ],
                'common_flaws': {
                    'weight_distribution': 'Ensure proper weight transfer to your front foot - staying back limits your options',
                    'bat_path': 'Keep your bat coming down straight - angled bat paths cause edges',
                    'head_stability': 'Maintain head position throughout the shot - any movement disrupts timing',
                    'planting_front_foot': 'Let your head lead, not your foot. Pre-planting limits adjustment',
                    'stiff_front_leg': 'Bent front knee absorbs impact and provides stability',
                    'back_foot_lifting': 'Keep back foot grounded for balance until shot is complete'
                },
                'drills': [
                    'Front foot stride practice with marker cones',
                    'Head position drills - balance book on head during shadow batting',
                    'Weight transfer exercises with resistance bands',
                    'Slow motion practice focusing on sequence of movements'
                ]
            },
            'FLICK': {
                'description': 'The flick or clip is a wristy shot played to deliveries on the pads, directing the ball to the leg side. It requires excellent timing, supple wrists, and precise bat control. The shot can be played off front or back foot and is essential for scoring against straight bowling.',
                'key_focus': [
                    'Wrist rotation at point of contact',
                    'Meet ball in front of pad',
                    'Soft hands for control',
                    'Head still over the ball',
                    'Front foot clearing the way',
                    'Bottom hand guides placement',
                    'Watch ball onto bat',
                    'Controlled follow-through'
                ],
                'common_flaws': {
                    'hitting_across': 'Play along the line first, then turn wrists. Hitting across leads to LBW',
                    'hard_hands': 'Flicks need soft, supple wrists. Tense grip kills the shot',
                    'closing_face_early': 'Wait until impact to turn bat face. Early closing mishits',
                    'falling_over': 'Stay balanced - don\'t fall to leg side',
                    'playing_too_early': 'Wait for ball to come to you. Playing early hits pad'
                },
                'drills': [
                    'Wrist flexibility exercises with bat',
                    'Throw-downs on leg stump - practice placement',
                    'Flick vs defend decision drills',
                    'One-handed bottom hand practice for feel'
                ]
            },
            'LEAVE': {
                'description': 'Leaving the ball is as important as playing it. Good judgment of line and length, combined with proper technique when shouldering arms, protects your wicket and frustrates bowlers. The leave is a statement of control and composure.',
                'key_focus': [
                    'Early judgment of line and length',
                    'Bat raised out of way',
                    'Body balanced and still',
                    'Eyes following ball past',
                    'Back foot movement if needed',
                    'Confident, decisive movement',
                    'Arms away from body',
                    'Relaxed shoulders'
                ],
                'common_flaws': {
                    'late_decision': 'Decide early to leave. Late decisions cause half-shots',
                    'bat_in_way': 'Get bat completely clear. Half-leaves hit the edge',
                    'body_in_line': 'Move inside line of ball if it\'s doing a lot. Getting hit hurts',
                    'not_watching': 'Watch ball all the way. Builds understanding of conditions',
                    'tense_shoulders': 'Stay relaxed when leaving. Tension affects next ball'
                },
                'drills': [
                    'Leave practice with colored balls - leave red, play blue',
                    'Line and length judgment from side on video',
                    'Practice sessions where you can only leave - builds patience',
                    'Shoulder arms technique in front of mirror'
                ]
            },
            'BACK FOOT DEFENSE': {
                'description': 'The back foot defensive shot is played to short-of-length deliveries that don\'t warrant an attacking stroke. It\'s about getting behind the line, playing late, and controlling the ball down. Essential against pace and bounce.',
                'key_focus': [
                    'Quick back foot movement',
                    'Get behind the line of ball',
                    'High hands, soft grip',
                    'Play under the eyes',
                    'Angled bat to play down',
                    'Weight balanced on both feet',
                    'Watch ball onto bat',
                    'Controlled follow-through down'
                ],
                'common_flaws': {
                    'playing_away_from_body': 'Play under your eyes. Reaching for ball creates edges',
                    'hard_hands': 'Soft hands kill the pace and prevent catches',
                    'flat_bat': 'Angle bat down to cover bounce. Flat bat pops up catches',
                    'falling_over': 'Stay balanced on both feet. Don\'t fall to off side',
                    'not_getting_back': 'Full back and across movement. Half movements leave gaps'
                },
                'drills': [
                    'Back foot trigger movement practice',
                    'Short ball defense with tennis balls',
                    'Shadow batting focusing on high hands',
                    'Catching practice to develop soft hands'
                ]
            }
        }

    def generate_advice(self, metrics: Dict) -> Dict:
        """
        Generate comprehensive, shot-specific coaching advice

        Args:
            metrics: Dictionary containing all calculated metrics including shot_type

        Returns:
            Dictionary with strengths, flaws, recommendations, and shot-specific insights
        """
        advice = {
            'shot_type': metrics.get('shot_type', 'UNKNOWN'),
            'shot_confidence': metrics.get('shot_confidence', 0.0),
            'strengths': [],
            'flaws': [],
            'recommendations': [],
            'shot_insights': {}
        }

        shot_type = metrics.get('shot_type', 'UNKNOWN')

        # Add shot-specific context
        if shot_type in self.shot_specific_advice:
            shot_data = self.shot_specific_advice[shot_type]
            advice['shot_insights'] = {
                'description': shot_data['description'],
                'key_focus_areas': shot_data['key_focus'],
                'recommended_drills': shot_data.get('drills', [])
            }

        # Analyze metrics with shot-specific context
        metric_analysis = self._analyze_metrics(metrics, shot_type)

        # Generate strengths
        strengths_count = 0
        for metric, analysis in metric_analysis.items():
            if analysis['status'] == 'excellent' and strengths_count < 3:  # Limit to top 3 strengths
                advice['strengths'].append(self._get_contextual_feedback(metric, analysis, shot_type, 'strength'))
                strengths_count += 1

        # Generate flaws with shot-specific context
        flaws_count = 0
        priority_flaws = []
        for metric, analysis in metric_analysis.items():
            if analysis['status'] in ['poor', 'needs_improvement']:
                priority = 'high' if analysis['status'] == 'poor' else 'medium'
                priority_flaws.append((metric, analysis, priority))

        # Sort by priority and limit to top 3-4 flaws
        priority_flaws.sort(key=lambda x: 0 if x[2] == 'high' else 1)
        for metric, analysis, priority in priority_flaws[:4]:
            advice['flaws'].append(self._get_contextual_feedback(metric, analysis, shot_type, 'flaw'))

        # Generate actionable recommendations
        for metric, analysis, priority in priority_flaws[:3]:  # Top 3 issues
            recommendation = self._get_contextual_feedback(metric, analysis, shot_type, 'recommendation')
            if recommendation:
                advice['recommendations'].append(recommendation)

        # Add shot-specific recommendations if applicable
        shot_specific_rec = self._get_shot_specific_recommendations(metrics, shot_type)
        if shot_specific_rec:
            advice['recommendations'].extend(shot_specific_rec[:2])  # Add up to 2 shot-specific tips

        # Ensure we have at least one item in each category
        if not advice['strengths']:
            advice['strengths'].append("✅ Good attempt - continue practicing to refine your technique")

        if not advice['flaws']:
            advice['flaws'].append("✅ Solid technique overall - minor refinements will take you to the next level")

        if not advice['recommendations']:
            advice['recommendations'].append("💡 Keep practicing your current technique with focus on consistency")

        return advice

    def _analyze_metrics(self, metrics: Dict, shot_type: str) -> Dict:
        """Analyze metrics with shot-specific context"""
        analysis = {}

        # Elbow angle analysis
        if 'max_elbow_angle' in metrics:
            angle = metrics['max_elbow_angle']
            # Drives and defensive shots require straighter arms
            if shot_type in ['DRIVE', 'DEFENSIVE', 'FORWARD SHOT']:
                if angle >= 165:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'excellent', 'importance': 'high'}
                elif angle >= 155:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'needs_improvement', 'importance': 'high'}
                else:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'poor', 'importance': 'high'}
            else:
                # More flexible for pull/cut/sweep
                if angle >= 150:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'excellent', 'importance': 'medium'}
                elif angle >= 135:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'needs_improvement', 'importance': 'medium'}
                else:
                    analysis['elbow_angle'] = {'value': angle, 'status': 'poor', 'importance': 'high'}

        # Weight transfer analysis (shot-specific)
        if 'weight_transfer_amount' in metrics:
            transfer = metrics['weight_transfer_amount']

            if shot_type in ['DRIVE', 'FORWARD SHOT', 'LOFTED']:
                # Forward shots need positive weight transfer
                if transfer >= 8:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'excellent', 'importance': 'high'}
                elif transfer >= 3:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'needs_improvement', 'importance': 'high'}
                else:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'poor', 'importance': 'high'}

            elif shot_type in ['PULL/HOOK', 'CUT']:
                # Back foot shots should have minimal or negative weight transfer
                if transfer <= 2:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'excellent', 'importance': 'high'}
                elif transfer <= 5:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'needs_improvement', 'importance': 'high'}
                else:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'poor', 'importance': 'high'}

            else:
                # Defensive and sweep - moderate
                if 0 <= transfer <= 5:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'excellent', 'importance': 'medium'}
                else:
                    analysis['weight_transfer'] = {'value': transfer, 'status': 'needs_improvement', 'importance': 'medium'}

        # Body rotation analysis
        if 'body_rotation_total' in metrics:
            rotation = metrics['body_rotation_total']

            if shot_type in ['PULL/HOOK', 'SWEEP', 'CUT']:
                # These shots need more rotation
                if rotation >= 45:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'excellent', 'importance': 'high'}
                elif rotation >= 30:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'needs_improvement', 'importance': 'high'}
                else:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'poor', 'importance': 'high'}

            elif shot_type == 'DEFENSIVE':
                # Defensive shots need minimal rotation
                if rotation <= 15:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'excellent', 'importance': 'high'}
                elif rotation <= 25:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'needs_improvement', 'importance': 'medium'}
                else:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'poor', 'importance': 'medium'}

            else:
                # Forward shots - moderate rotation
                if 20 <= rotation <= 45:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'excellent', 'importance': 'medium'}
                elif rotation < 15 or rotation > 60:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'poor', 'importance': 'high'}
                else:
                    analysis['body_rotation'] = {'value': rotation, 'status': 'needs_improvement', 'importance': 'medium'}

        # Head stability analysis (critical for all shots)
        if 'head_movement' in metrics:
            head_movement = metrics['head_movement']
            if head_movement <= 8:
                analysis['head_stability'] = {'value': head_movement, 'status': 'excellent', 'importance': 'high'}
            elif head_movement <= 20:
                analysis['head_stability'] = {'value': head_movement, 'status': 'needs_improvement', 'importance': 'high'}
            else:
                analysis['head_stability'] = {'value': head_movement, 'status': 'poor', 'importance': 'high'}

        # Knee analysis
        if 'knee_bracing' in metrics:
            knee_bracing = metrics['knee_bracing']
            if shot_type in ['DRIVE', 'FORWARD SHOT']:
                # Forward shots need good front leg bracing
                if knee_bracing >= 15:
                    analysis['knee_bracing'] = {'value': knee_bracing, 'status': 'excellent', 'importance': 'medium'}
                elif knee_bracing >= 5:
                    analysis['knee_bracing'] = {'value': knee_bracing, 'status': 'needs_improvement', 'importance': 'medium'}
                else:
                    analysis['knee_bracing'] = {'value': knee_bracing, 'status': 'poor', 'importance': 'medium'}

        if 'stance_knee_angle' in metrics:
            knee_angle = metrics['stance_knee_angle']
            if 120 <= knee_angle <= 145:
                analysis['stance_knee'] = {'value': knee_angle, 'status': 'excellent', 'importance': 'low'}
            elif knee_angle < 110 or knee_angle > 160:
                analysis['stance_knee'] = {'value': knee_angle, 'status': 'poor', 'importance': 'medium'}
            else:
                analysis['stance_knee'] = {'value': knee_angle, 'status': 'needs_improvement', 'importance': 'low'}

        return analysis

    def _get_contextual_feedback(self, metric: str, analysis: Dict, shot_type: str, feedback_type: str) -> str:
        """Generate contextual feedback based on metric, shot type, and feedback category"""
        value = analysis['value']
        status = analysis['status']

        if feedback_type == 'strength':
            if metric == 'elbow_angle':
                return f"✅ Excellent arm extension ({value:.1f}°) - your front arm is beautifully straight, generating optimal power and control"
            elif metric == 'weight_transfer':
                if shot_type in ['DRIVE', 'FORWARD SHOT']:
                    return f"✅ Perfect weight transfer forward ({value:.1f}) - you're getting excellent momentum into the shot"
                elif shot_type in ['PULL/HOOK', 'CUT']:
                    return f"✅ Good weight distribution ({value:.1f}) - well balanced for a back-foot shot"
            elif metric == 'body_rotation':
                if shot_type in ['PULL/HOOK', 'SWEEP']:
                    return f"✅ Powerful body rotation ({value:.1f}°) - great use of your core to generate power"
                elif shot_type == 'DEFENSIVE':
                    return f"✅ Controlled body movement ({value:.1f}°) - excellent defensive technique with minimal rotation"
                else:
                    return f"✅ Good shoulder turn ({value:.1f}°) - optimal rotation for this shot"
            elif metric == 'head_stability':
                return f"✅ Excellent head stability ({value:.1f}° movement) - your head position is rock solid, crucial for timing"
            elif metric == 'knee_bracing':
                return f"✅ Strong front leg bracing ({value:.1f}° extension) - solid base for power transfer"
            elif metric == 'stance_knee':
                return f"✅ Balanced stance position ({value:.1f}°) - good athletic base"

        elif feedback_type == 'flaw':
            if metric == 'elbow_angle':
                if shot_type in ['DRIVE', 'FORWARD SHOT']:
                    return f"⚠️ Bent front arm ({value:.1f}°) - for drives, keep your front arm straighter through impact for better timing and power"
                else:
                    return f"⚠️ Elbow too bent ({value:.1f}°) - work on fuller arm extension to improve bat speed"
            elif metric == 'weight_transfer':
                if shot_type in ['DRIVE', 'FORWARD SHOT', 'LOFTED']:
                    if value < 0:
                        return f"⚠️ Weight going backwards ({value:.1f}) - you're falling away from the ball. Get your weight moving forward"
                    else:
                        return f"⚠️ Insufficient weight transfer ({value:.1f}) - commit more to moving forward into the shot"
                elif shot_type in ['PULL/HOOK', 'CUT']:
                    return f"⚠️ Too much forward weight transfer ({value:.1f}) - these back-foot shots need weight back, not forward"
            elif metric == 'body_rotation':
                if shot_type in ['PULL/HOOK', 'CUT']:
                    return f"⚠️ Limited body rotation ({value:.1f}°) - generate more power by using your hips and shoulders"
                elif shot_type == 'DEFENSIVE':
                    return f"⚠️ Excessive body movement ({value:.1f}°) - defensive shots need minimal rotation for better control"
                else:
                    if value < 15:
                        return f"⚠️ Too little body rotation ({value:.1f}°) - engage your core more to generate power"
                    else:
                        return f"⚠️ Over-rotating ({value:.1f}°) - this can cause loss of balance and power"
            elif metric == 'head_stability':
                return f"⚠️ Head movement detected ({value:.1f}°) - excessive head movement disrupts timing and ball tracking"
            elif metric == 'knee_bracing':
                return f"⚠️ Weak front leg ({value:.1f}° change) - brace your front leg more firmly for better energy transfer"
            elif metric == 'stance_knee':
                if value < 120:
                    return f"⚠️ Stance too low ({value:.1f}°) - raise your stance slightly for better movement"
                else:
                    return f"⚠️ Stance too upright ({value:.1f}°) - lower your stance for better balance"

        elif feedback_type == 'recommendation':
            if metric == 'elbow_angle':
                return "💡 Drill: Practice shadow batting with emphasis on keeping your front arm fully extended. Hold the finish position to feel the stretch"
            elif metric == 'weight_transfer':
                if shot_type in ['DRIVE', 'FORWARD SHOT']:
                    return "💡 Drill: Practice stepping forward into a front-foot drive, focusing on feeling your weight shift onto your front foot"
                elif shot_type in ['PULL/HOOK', 'CUT']:
                    return "💡 Drill: Practice weight transfer to your back foot. Do shadow pulls, ensuring you feel balanced on your back leg"
            elif metric == 'body_rotation':
                if shot_type in ['PULL/HOOK']:
                    return "💡 Drill: Practice hip and shoulder rotation exercises. Focus on pivoting your back foot to generate power"
                elif shot_type == 'DEFENSIVE':
                    return "💡 Drill: Practice defensive shots with focus on minimal body movement - quiet hands and solid base"
                else:
                    return "💡 Drill: Work on core rotation - practice turning your shoulders while keeping head still"
            elif metric == 'head_stability':
                return "💡 Drill: Place a cap with water on your head during shadow batting. Practice keeping it balanced - this forces head stillness"
            elif metric == 'knee_bracing':
                return "💡 Drill: Practice leg strengthening exercises (lunges, single-leg squats) to build a stronger, more stable base"

        return ""

    def _get_shot_specific_recommendations(self, metrics: Dict, shot_type: str) -> List[str]:
        """Get additional shot-specific coaching tips"""
        recommendations = []

        if shot_type not in self.shot_specific_advice:
            return recommendations

        shot_info = self.shot_specific_advice[shot_type]

        # Add pro tips for specific shot types
        pro_tips = {
            'DRIVE': "💡 Pro Tip: For classic drives, imagine painting a straight line from your bat's position at address to your follow-through. Your head should be the heaviest thing going forward.",
            'PULL/HOOK': "💡 Pro Tip: Watch the ball onto the bat - pull shots require excellent ball tracking. Commit early but execute late, and roll your wrists at impact to keep the ball down.",
            'CUT': "💡 Pro Tip: Use your bottom hand to guide and your top hand to control - wrist flexibility is key. Wait for the ball to come to you, don't go looking for it.",
            'DEFENSIVE': "💡 Pro Tip: Think 'soft hands' - defensive shots should deaden the ball, not push it. Imagine you're catching an egg - that's the grip pressure you need.",
            'SWEEP': "💡 Pro Tip: Get your front pad outside the line to give yourself room and protection. The sweep is premeditated - decide early and commit fully.",
            'LOFTED': "💡 Pro Tip: Trust your technique and swing through - don't try to hit too hard, let timing do the work. Get to the pitch of the ball with your feet.",
            'FORWARD SHOT': "💡 Pro Tip: Your head should lead every forward movement. Where your head goes, your body follows. Practice until it becomes automatic.",
            'FLICK': "💡 Pro Tip: The flick is all about timing the wrist turn. Play along the line first, then roll the wrists - think of it as redirecting, not hitting.",
            'LEAVE': "💡 Pro Tip: The best batsmen know what not to play. A good leave is as valuable as a good shot - it shows control and understanding of your off stump.",
            'BACK FOOT DEFENSE': "💡 Pro Tip: Get back and across quickly, but play the ball under your eyes. High hands and soft grip are your best friends against pace."
        }

        if shot_type in pro_tips:
            recommendations.append(pro_tips[shot_type])

        # Add a relevant drill from the database
        if 'drills' in shot_info and shot_info['drills']:
            drill = random.choice(shot_info['drills'])
            recommendations.append(f"🏋️ Recommended Drill: {drill}")

        return recommendations

    def get_advice_summary(self, advice: Dict) -> str:
        """Generate a formatted summary for export/sharing"""
        lines = []

        lines.append("=" * 60)
        lines.append("CRICKET COACHING ANALYSIS REPORT")
        lines.append("=" * 60)

        if advice.get('shot_type') and advice['shot_type'] != 'UNKNOWN':
            lines.append(f"\n🏏 Shot Type Detected: {advice['shot_type']} (Confidence: {advice.get('shot_confidence', 0)*100:.0f}%)")
            if 'shot_insights' in advice and 'description' in advice['shot_insights']:
                lines.append(f"   {advice['shot_insights']['description']}")

        if advice['strengths']:
            lines.append("\n✅ STRENGTHS:")
            for strength in advice['strengths']:
                lines.append(f"   {strength}")

        if advice['flaws']:
            lines.append("\n⚠️  AREAS FOR IMPROVEMENT:")
            for flaw in advice['flaws']:
                lines.append(f"   {flaw}")

        if advice['recommendations']:
            lines.append("\n💡 ACTIONABLE RECOMMENDATIONS:")
            for i, rec in enumerate(advice['recommendations'], 1):
                lines.append(f"   {i}. {rec}")

        if 'shot_insights' in advice and 'key_focus_areas' in advice['shot_insights']:
            lines.append(f"\n🎯 KEY FOCUS AREAS FOR {advice['shot_type']}:")
            for focus in advice['shot_insights']['key_focus_areas'][:6]:  # Show top 6
                lines.append(f"   • {focus}")

        if 'shot_insights' in advice and 'recommended_drills' in advice['shot_insights']:
            drills = advice['shot_insights']['recommended_drills']
            if drills:
                lines.append(f"\n🏋️ PRACTICE DRILLS:")
                for drill in drills[:3]:  # Show top 3 drills
                    lines.append(f"   • {drill}")

        lines.append("\n" + "=" * 60)
        lines.append("Keep practicing - consistency is the key to mastery!")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_llm_narrative(self, advice: Dict, metrics: Dict) -> str:
        """
        Generate a personalized coaching narrative using OpenAI GPT.
        Falls back to basic summary if API is unavailable.
        """
        client = get_openai_client()

        if client is None:
            return self._generate_fallback_narrative(advice)

        try:
            # Build context for the LLM
            shot_type = advice.get('shot_type', 'UNKNOWN')
            strengths = advice.get('strengths', [])
            flaws = advice.get('flaws', [])

            prompt = f"""You are an experienced cricket batting coach providing personalized feedback to a player.

Based on the video analysis of their {shot_type}, here's what was detected:

**Strengths identified:**
{chr(10).join(f"- {s}" for s in strengths) if strengths else "- No major strengths detected"}

**Areas needing improvement:**
{chr(10).join(f"- {f}" for f in flaws) if flaws else "- No major issues detected"}

**Key metrics:**
- Elbow angle: {metrics.get('max_elbow_angle', 0):.1f}°
- Weight transfer: {metrics.get('weight_transfer_amount', 0):.1f}%
- Body rotation: {metrics.get('body_rotation_total', 0):.1f}°
- Head movement: {metrics.get('head_movement', 0):.1f}°

Write a 3-4 sentence personalized coaching summary that:
1. Acknowledges what they did well (be specific)
2. Identifies the most important thing to work on
3. Gives one actionable tip they can try in their next session
4. Sounds encouraging but honest

Keep it conversational and cricket-specific. Reference professional players if relevant."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert cricket batting coach with decades of experience coaching at all levels."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_narrative(advice)

    def _generate_fallback_narrative(self, advice: Dict) -> str:
        """Generate a basic narrative when LLM is unavailable"""
        shot_type = advice.get('shot_type', 'shot')

        if advice.get('strengths') and len(advice['strengths']) > 0:
            strength_text = "Your technique shows good fundamentals"
        else:
            strength_text = "Keep working on the basics"

        if advice.get('flaws') and len(advice['flaws']) > 0:
            focus_text = "Focus on the areas highlighted above for improvement"
        else:
            focus_text = "Continue refining your current technique"

        return f"Good effort on your {shot_type}. {strength_text}. {focus_text}. Consistent practice will help you develop muscle memory and confidence."

    def generate_ai_advice(self, metrics: Dict) -> Dict:
        """Generate comprehensive AI advice - wrapper for generate_advice"""
        return self.generate_advice(metrics)


class ImprovementEngine:
    """
    Analyzes improvement between original and follow-up cricket shot attempts.
    Provides honest feedback - if the user didn't improve, it will say so.
    """

    def __init__(self):
        # Optimal ranges for each metric by shot type
        self.optimal_ranges = {
            'DRIVE': {
                'max_elbow_angle': (165, 180),
                'weight_transfer_amount': (8, 20),
                'body_rotation_total': (20, 45),
                'head_movement': (0, 10)
            },
            'PULL/HOOK': {
                'max_elbow_angle': (150, 180),
                'weight_transfer_amount': (-5, 2),
                'body_rotation_total': (45, 75),
                'head_movement': (0, 12)
            },
            'CUT': {
                'max_elbow_angle': (150, 180),
                'weight_transfer_amount': (-3, 5),
                'body_rotation_total': (45, 70),
                'head_movement': (0, 12)
            },
            'DEFENSIVE': {
                'max_elbow_angle': (160, 180),
                'weight_transfer_amount': (0, 5),
                'body_rotation_total': (0, 15),
                'head_movement': (0, 8)
            },
            'SWEEP': {
                'max_elbow_angle': (140, 170),
                'weight_transfer_amount': (3, 15),
                'body_rotation_total': (50, 80),
                'head_movement': (0, 15)
            },
            'LOFTED': {
                'max_elbow_angle': (155, 180),
                'weight_transfer_amount': (5, 20),
                'body_rotation_total': (25, 55),
                'head_movement': (0, 12)
            },
            'FORWARD SHOT': {
                'max_elbow_angle': (160, 180),
                'weight_transfer_amount': (5, 15),
                'body_rotation_total': (15, 40),
                'head_movement': (0, 10)
            }
        }

        # Default optimal ranges
        self.default_optimal = {
            'max_elbow_angle': (155, 180),
            'weight_transfer_amount': (3, 15),
            'body_rotation_total': (20, 50),
            'head_movement': (0, 12)
        }

        # Metric display names and units
        self.metric_info = {
            'max_elbow_angle': {'name': 'Elbow Extension', 'unit': '°', 'higher_is_better': True},
            'weight_transfer_amount': {'name': 'Weight Transfer', 'unit': '%', 'higher_is_better': None},  # Depends on shot
            'body_rotation_total': {'name': 'Body Rotation', 'unit': '°', 'higher_is_better': None},  # Depends on shot
            'head_movement': {'name': 'Head Stability', 'unit': '°', 'higher_is_better': False}  # Lower is better
        }

    def _get_optimal_range(self, metric: str, shot_type: str) -> Tuple[float, float]:
        """Get the optimal range for a metric based on shot type"""
        shot_ranges = self.optimal_ranges.get(shot_type, self.default_optimal)
        return shot_ranges.get(metric, self.default_optimal.get(metric, (0, 100)))

    def _calculate_metric_score(self, value: float, optimal_range: Tuple[float, float],
                                 higher_is_better: bool = None) -> float:
        """
        Calculate how close a metric value is to optimal (0-100 score).
        """
        opt_min, opt_max = optimal_range
        opt_mid = (opt_min + opt_max) / 2

        # If value is within optimal range, score is 80-100
        if opt_min <= value <= opt_max:
            # Perfect score if at midpoint, 80 at edges
            distance_from_mid = abs(value - opt_mid)
            range_half = (opt_max - opt_min) / 2
            if range_half > 0:
                return 100 - (distance_from_mid / range_half) * 20
            return 100

        # Outside optimal range - calculate penalty
        if value < opt_min:
            distance = opt_min - value
            penalty = min(distance * 2, 50)  # Max 50 point penalty
            return max(80 - penalty, 20)
        else:  # value > opt_max
            distance = value - opt_max
            penalty = min(distance * 2, 50)
            return max(80 - penalty, 20)

    def _calculate_improvement_delta(self, original: float, followup: float,
                                     optimal_range: Tuple[float, float],
                                     metric: str, shot_type: str) -> Dict:
        """
        Calculate the improvement between original and follow-up values.
        Returns improvement details including direction and magnitude.
        """
        opt_min, opt_max = optimal_range
        opt_mid = (opt_min + opt_max) / 2

        # Calculate distance from optimal midpoint
        original_distance = abs(original - opt_mid)
        followup_distance = abs(followup - opt_mid)

        # Calculate scores for both
        original_score = self._calculate_metric_score(original, optimal_range)
        followup_score = self._calculate_metric_score(followup, optimal_range)

        # Determine improvement
        score_change = followup_score - original_score

        # Determine improvement status
        if score_change > 5:
            status = 'improved'
        elif score_change < -5:
            status = 'regressed'
        else:
            status = 'maintained'

        return {
            'original_value': original,
            'followup_value': followup,
            'original_score': original_score,
            'followup_score': followup_score,
            'score_change': score_change,
            'status': status,
            'optimal_range': optimal_range
        }

    def analyze_improvement(self, original_metrics: Dict, followup_metrics: Dict) -> Dict:
        """
        Main function to analyze improvement between two shot attempts.
        Returns comprehensive comparison with honest assessment.
        """
        shot_type = original_metrics.get('shot_type', 'FORWARD SHOT')
        followup_shot_type = followup_metrics.get('shot_type', 'FORWARD SHOT')

        # Key metrics to compare
        comparison_metrics = ['max_elbow_angle', 'weight_transfer_amount',
                            'body_rotation_total', 'head_movement']

        metric_comparisons = {}
        total_improvement_score = 0
        metrics_analyzed = 0

        improved_areas = []
        regressed_areas = []
        maintained_areas = []

        for metric in comparison_metrics:
            if metric in original_metrics and metric in followup_metrics:
                original_val = original_metrics[metric]
                followup_val = followup_metrics[metric]

                # Get optimal range for this shot type
                optimal_range = self._get_optimal_range(metric, shot_type)

                # Calculate improvement
                comparison = self._calculate_improvement_delta(
                    original_val, followup_val, optimal_range, metric, shot_type
                )

                metric_comparisons[metric] = comparison
                total_improvement_score += comparison['score_change']
                metrics_analyzed += 1

                # Categorize
                metric_name = self.metric_info[metric]['name']
                if comparison['status'] == 'improved':
                    improved_areas.append({
                        'metric': metric,
                        'name': metric_name,
                        'change': comparison['score_change'],
                        'from_value': original_val,
                        'to_value': followup_val,
                        'unit': self.metric_info[metric]['unit']
                    })
                elif comparison['status'] == 'regressed':
                    regressed_areas.append({
                        'metric': metric,
                        'name': metric_name,
                        'change': comparison['score_change'],
                        'from_value': original_val,
                        'to_value': followup_val,
                        'unit': self.metric_info[metric]['unit']
                    })
                else:
                    maintained_areas.append({
                        'metric': metric,
                        'name': metric_name,
                        'from_value': original_val,
                        'to_value': followup_val,
                        'unit': self.metric_info[metric]['unit']
                    })

        # Calculate overall improvement score (0-100)
        avg_improvement = total_improvement_score / max(metrics_analyzed, 1)

        # Calculate accuracy scores for both attempts
        original_accuracy = self._calculate_overall_accuracy(original_metrics, shot_type)
        followup_accuracy = self._calculate_overall_accuracy(followup_metrics, shot_type)

        # Determine overall verdict - HONEST ASSESSMENT
        if followup_accuracy > original_accuracy + 5:
            overall_verdict = 'IMPROVED'
            verdict_description = 'Great progress! Your technique has measurably improved.'
        elif followup_accuracy < original_accuracy - 5:
            overall_verdict = 'REGRESSED'
            verdict_description = 'Your follow-up attempt shows some regression. This can happen when trying new techniques - keep practicing.'
        else:
            if len(improved_areas) > len(regressed_areas):
                overall_verdict = 'SLIGHT_IMPROVEMENT'
                verdict_description = 'Marginal improvement detected. You\'re on the right track but there\'s more work to do.'
            elif len(regressed_areas) > len(improved_areas):
                overall_verdict = 'SLIGHT_REGRESSION'
                verdict_description = 'Your technique has slightly regressed. Review the feedback from your first attempt and focus on those areas.'
            else:
                overall_verdict = 'MAINTAINED'
                verdict_description = 'Your technique is consistent between attempts. Focus on specific areas to see improvement.'

        return {
            'shot_type': shot_type,
            'followup_shot_type': followup_shot_type,
            'metric_comparisons': metric_comparisons,
            'improved_areas': improved_areas,
            'regressed_areas': regressed_areas,
            'maintained_areas': maintained_areas,
            'original_accuracy': original_accuracy,
            'followup_accuracy': followup_accuracy,
            'accuracy_change': followup_accuracy - original_accuracy,
            'overall_verdict': overall_verdict,
            'verdict_description': verdict_description,
            'improvement_score': min(max(50 + avg_improvement, 0), 100)  # Centered at 50
        }

    def _calculate_overall_accuracy(self, metrics: Dict, shot_type: str) -> float:
        """Calculate overall accuracy score (0-100) for a shot attempt"""
        scores = []

        for metric in ['max_elbow_angle', 'weight_transfer_amount',
                      'body_rotation_total', 'head_movement']:
            if metric in metrics:
                optimal_range = self._get_optimal_range(metric, shot_type)
                score = self._calculate_metric_score(metrics[metric], optimal_range)
                scores.append(score)

        return round(sum(scores) / max(len(scores), 1), 1)

    def generate_improvement_feedback(self, analysis: Dict) -> Dict:
        """
        Generate detailed feedback based on the improvement analysis.
        """
        feedback = {
            'summary': '',
            'improvements': [],
            'regressions': [],
            'focus_areas': [],
            'drills': []
        }

        verdict = analysis['overall_verdict']

        # Generate summary based on verdict
        if verdict == 'IMPROVED':
            feedback['summary'] = (
                f"Excellent work! Your accuracy improved from {analysis['original_accuracy']:.0f}/100 to "
                f"{analysis['followup_accuracy']:.0f}/100. Your dedication to practice is paying off."
            )
        elif verdict == 'REGRESSED':
            feedback['summary'] = (
                f"Your accuracy dropped from {analysis['original_accuracy']:.0f}/100 to "
                f"{analysis['followup_accuracy']:.0f}/100. Don't be discouraged - this often happens "
                f"when making technical changes. Review the original advice and try again."
            )
        elif verdict == 'SLIGHT_IMPROVEMENT':
            feedback['summary'] = (
                f"You're making progress! Accuracy moved from {analysis['original_accuracy']:.0f}/100 to "
                f"{analysis['followup_accuracy']:.0f}/100. Keep working on the areas highlighted below."
            )
        elif verdict == 'SLIGHT_REGRESSION':
            feedback['summary'] = (
                f"Your accuracy dipped slightly from {analysis['original_accuracy']:.0f}/100 to "
                f"{analysis['followup_accuracy']:.0f}/100. Focus on the fundamentals and try again."
            )
        else:  # MAINTAINED
            feedback['summary'] = (
                f"Your technique is consistent at {analysis['followup_accuracy']:.0f}/100. "
                f"To see improvement, focus specifically on the areas marked for improvement."
            )

        # Generate improvement feedback
        for area in analysis['improved_areas']:
            feedback['improvements'].append(
                f"✅ {area['name']}: Improved from {area['from_value']:.1f}{area['unit']} to "
                f"{area['to_value']:.1f}{area['unit']} (+{area['change']:.0f} points)"
            )

        # Generate regression feedback (HONEST)
        for area in analysis['regressed_areas']:
            feedback['regressions'].append(
                f"⚠️ {area['name']}: Regressed from {area['from_value']:.1f}{area['unit']} to "
                f"{area['to_value']:.1f}{area['unit']} ({area['change']:.0f} points)"
            )

        # Generate focus areas based on what needs work
        focus_recommendations = self._get_focus_recommendations(analysis)
        feedback['focus_areas'] = focus_recommendations

        # Generate drill recommendations
        drills = self._get_improvement_drills(analysis)
        feedback['drills'] = drills

        return feedback

    def _get_focus_recommendations(self, analysis: Dict) -> List[str]:
        """Get specific focus recommendations based on analysis"""
        recommendations = []
        shot_type = analysis['shot_type']

        # Prioritize regressed areas
        for area in analysis['regressed_areas']:
            metric = area['metric']
            if metric == 'max_elbow_angle':
                recommendations.append(
                    "🎯 Focus on elbow extension - keep your front arm straighter through the shot"
                )
            elif metric == 'weight_transfer_amount':
                if shot_type in ['DRIVE', 'FORWARD SHOT', 'LOFTED']:
                    recommendations.append(
                        "🎯 Work on weight transfer - commit more to moving forward into the shot"
                    )
                else:
                    recommendations.append(
                        "🎯 Adjust weight distribution - for this shot, stay more balanced or back"
                    )
            elif metric == 'body_rotation_total':
                if shot_type in ['PULL/HOOK', 'CUT', 'SWEEP']:
                    recommendations.append(
                        "🎯 Generate more power through hip and shoulder rotation"
                    )
                elif shot_type == 'DEFENSIVE':
                    recommendations.append(
                        "🎯 Minimize body rotation - defensive shots need stillness"
                    )
                else:
                    recommendations.append(
                        "🎯 Optimize body rotation for better power and control"
                    )
            elif metric == 'head_movement':
                recommendations.append(
                    "🎯 Keep your head still - excessive movement disrupts timing"
                )

        # Add general recommendations if nothing specific
        if not recommendations:
            if analysis['overall_verdict'] in ['MAINTAINED', 'SLIGHT_REGRESSION']:
                recommendations.append(
                    "🎯 Focus on consistency - practice the same shot repeatedly with attention to form"
                )
            recommendations.append(
                "🎯 Continue working on timing and balance throughout your shot"
            )

        return recommendations[:3]  # Limit to 3

    def _get_improvement_drills(self, analysis: Dict) -> List[str]:
        """Get drill recommendations for continued improvement"""
        drills = []
        shot_type = analysis['shot_type']

        # Drill database by metric and shot type
        drill_database = {
            'max_elbow_angle': [
                "Shadow batting drill: Practice with focus on full arm extension at impact point",
                "Wall drill: Stand sideways to wall, practice extending arm without hitting wall",
                "Mirror work: Watch your elbow position throughout the swing"
            ],
            'weight_transfer_amount': {
                'forward': [
                    "Step and drive: Exaggerate stepping into the shot during practice",
                    "Single stump drill: Focus on driving through the line towards a single stump",
                    "Weighted bat practice: Builds strength for committed forward movement"
                ],
                'back': [
                    "Back foot punch drill: Practice quick weight shifts to back foot",
                    "Short ball reaction drill: Tennis ball bouncer practice for back foot shots"
                ]
            },
            'body_rotation_total': [
                "Hip rotation exercise: Practice rotating hips while keeping head still",
                "Core strengthening: Planks and rotational exercises improve power generation",
                "Resistance band rotation: Builds muscle memory for proper rotation"
            ],
            'head_movement': [
                "Balance book drill: Practice with object balanced on head (forces stillness)",
                "Eyes on ball drill: Track the ball with only your eyes, not your head",
                "Video analysis: Record yourself and check head position frame by frame"
            ]
        }

        # Add drills for regressed areas first
        for area in analysis['regressed_areas']:
            metric = area['metric']
            if metric == 'weight_transfer_amount':
                if shot_type in ['DRIVE', 'FORWARD SHOT', 'LOFTED']:
                    drills.extend(drill_database['weight_transfer_amount']['forward'][:1])
                else:
                    drills.extend(drill_database['weight_transfer_amount']['back'][:1])
            elif metric in drill_database:
                drills.append(drill_database[metric][0])

        # Add drills for maintained areas
        for area in analysis['maintained_areas']:
            metric = area['metric']
            if metric in drill_database and len(drills) < 3:
                if isinstance(drill_database[metric], list):
                    drills.append(drill_database[metric][1] if len(drill_database[metric]) > 1 else drill_database[metric][0])

        # Ensure at least 2 drills
        if len(drills) < 2:
            drills.append("General: Practice the shot 20-30 times focusing on one aspect at a time")
            drills.append("Video review: Record each session and compare to identify patterns")

        return drills[:4]  # Limit to 4 drills

    def get_improvement_summary(self, analysis: Dict, feedback: Dict) -> str:
        """Generate exportable text summary of improvement analysis"""
        lines = []

        lines.append("=" * 65)
        lines.append("CRICKET SHOT IMPROVEMENT ANALYSIS REPORT")
        lines.append("=" * 65)

        lines.append(f"\n📊 SHOT TYPE: {analysis['shot_type']}")

        lines.append(f"\n📈 ACCURACY COMPARISON:")
        lines.append(f"   Original Attempt:  {analysis['original_accuracy']:.0f}/100")
        lines.append(f"   Follow-up Attempt: {analysis['followup_accuracy']:.0f}/100")
        lines.append(f"   Change:            {analysis['accuracy_change']:+.0f} points")

        lines.append(f"\n🏆 VERDICT: {analysis['overall_verdict'].replace('_', ' ')}")
        lines.append(f"   {analysis['verdict_description']}")

        if feedback['improvements']:
            lines.append(f"\n✅ AREAS OF IMPROVEMENT:")
            for imp in feedback['improvements']:
                lines.append(f"   {imp}")

        if feedback['regressions']:
            lines.append(f"\n⚠️ AREAS NEEDING ATTENTION:")
            for reg in feedback['regressions']:
                lines.append(f"   {reg}")

        if feedback['focus_areas']:
            lines.append(f"\n🎯 FOCUS AREAS:")
            for focus in feedback['focus_areas']:
                lines.append(f"   {focus}")

        if feedback['drills']:
            lines.append(f"\n🏋️ RECOMMENDED DRILLS:")
            for i, drill in enumerate(feedback['drills'], 1):
                lines.append(f"   {i}. {drill}")

        lines.append("\n" + "=" * 65)
        lines.append("Keep practicing - improvement comes with consistent, focused effort!")
        lines.append("=" * 65)

        return "\n".join(lines)
