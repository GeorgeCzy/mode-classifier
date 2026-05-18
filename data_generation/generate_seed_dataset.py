"""Generate the first 500 seed examples for the response-mode classifier."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
JSONL_PATH = RAW_DIR / "seed_500.jsonl"
CSV_PATH = RAW_DIR / "seed_500.csv"

DATASET_VERSION = "seed_v1"
RANDOM_SEED = 20260518


CHAT_GROUPS = [
    (
        "greeting_small_talk",
        [
            "Hi, how are you today?",
            "Good morning, what can you help me with?",
            "It is nice to meet you.",
            "Are you having a good day?",
            "Hello there, who am I talking to?",
            "How should I address you?",
            "Can we chat for a minute?",
            "Tell me something interesting to start the day.",
            "What is your favorite way to greet people?",
            "Do you remember talking with me before?",
        ],
    ),
    (
        "identity_capabilities",
        [
            "What kind of robot are you?",
            "What are your main capabilities?",
            "Can you tell me what sensors you have?",
            "Do you understand spoken instructions?",
            "Are you able to learn my preferences?",
            "What tasks are you designed for?",
            "Do you need internet access to answer questions?",
            "Can you explain your limitations?",
            "How do you decide what to say?",
            "What languages can you communicate in?",
        ],
    ),
    (
        "factual_general_knowledge",
        [
            "What is the capital of Canada?",
            "Who wrote Pride and Prejudice?",
            "How far is the Moon from Earth?",
            "What is photosynthesis?",
            "When did the first airplane fly?",
            "What causes tides in the ocean?",
            "How many planets are in the solar system?",
            "What is the freezing point of water in Celsius?",
            "Who painted the Mona Lisa?",
            "What is a prime number?",
        ],
    ),
    (
        "explanation_learning",
        [
            "Can you explain recursion in simple terms?",
            "How does a microwave heat food?",
            "Why do leaves change color in autumn?",
            "Can you teach me the basics of probability?",
            "What is the difference between velocity and speed?",
            "How does a battery store energy?",
            "Explain machine learning without using jargon.",
            "Why is the sky blue?",
            "How do vaccines train the immune system?",
            "Can you summarize how GPS works?",
        ],
    ),
    (
        "recommendations_preferences",
        [
            "What book should I read if I like mystery stories?",
            "Can you suggest a calm playlist for studying?",
            "What is a healthy snack idea?",
            "Which movie would be good for a rainy evening?",
            "Can you recommend a beginner-friendly programming project?",
            "What should I cook with rice, eggs, and spinach?",
            "Which indoor plant is easy to care for?",
            "Can you suggest a simple workout plan I can read first?",
            "What podcast topic might help me learn history?",
            "What are some good ways to relax after work?",
        ],
    ),
    (
        "planning_scheduling",
        [
            "Can you help me plan my afternoon?",
            "What should I pack for a two-day trip?",
            "Help me organize a study schedule for next week.",
            "Can you make a checklist for cleaning the kitchen?",
            "How should I divide my time before the meeting?",
            "Please outline a morning routine for me.",
            "Can you help me prioritize these tasks?",
            "What is a sensible plan for preparing dinner?",
            "Can you draft a schedule for my exercise goals?",
            "How can I prepare for a visitor tomorrow?",
        ],
    ),
    (
        "reminders_information_only",
        [
            "Can you remind me what I said about the grocery list?",
            "What reminders did I ask for today?",
            "Tell me the steps for setting a reminder.",
            "What should I remember before leaving the house?",
            "Can you help me word a reminder to call Alex?",
            "What is a polite reminder message for my teammate?",
            "Can you list the deadlines I mentioned?",
            "How do I avoid forgetting my keys?",
            "What reminder phrase should I put on a sticky note?",
            "Can you summarize my errands so I remember them?",
        ],
    ),
    (
        "directions_navigation_verbal",
        [
            "Can you tell me how to get to the nearest elevator?",
            "Which hallway leads to the conference room?",
            "Describe the route to the cafeteria.",
            "Where is the exit from here?",
            "Can you explain where the charging station is?",
            "Which door should I use for the lab?",
            "How do I find the front desk?",
            "Can you give me walking directions to the library?",
            "Where is the restroom on this floor?",
            "Tell me which way the parking lot is.",
        ],
    ),
    (
        "emotional_support",
        [
            "I feel nervous about my presentation; what should I do?",
            "Can you say something encouraging?",
            "I had a rough day, can we talk?",
            "How do I calm down before an exam?",
            "Can you help me think through a conflict with a friend?",
            "I feel lonely right now.",
            "What is a kind message I can send myself?",
            "Can you help me reframe a mistake?",
            "What should I do when I feel overwhelmed?",
            "Can you listen while I vent?",
        ],
    ),
    (
        "creative_writing",
        [
            "Write a short poem about sunrise.",
            "Can you invent a name for my robot assistant?",
            "Give me a bedtime story idea.",
            "Write a friendly thank-you note.",
            "Can you create a slogan for a science fair booth?",
            "Make up a riddle about a clock.",
            "Draft a funny birthday message.",
            "Write a short scene set in a library.",
            "Can you help me brainstorm character names?",
            "Compose a haiku about rain.",
        ],
    ),
    (
        "translation_language",
        [
            "Translate 'good evening' into Spanish.",
            "How do I say thank you in Japanese?",
            "What does 'bonjour' mean?",
            "Can you correct this sentence: I has a dog?",
            "What is a more formal way to say 'help me out'?",
            "Explain the difference between affect and effect.",
            "Can you define the word 'resilient'?",
            "Give me three synonyms for 'quick'.",
            "How do I pronounce 'algorithm'?",
            "Can you make this sentence sound friendlier?",
        ],
    ),
    (
        "math_calculation",
        [
            "What is 18 percent of 250?",
            "Convert 3 miles to kilometers.",
            "How many minutes are in 2.5 hours?",
            "What is the square root of 144?",
            "If I split 42 cookies among 7 people, how many does each get?",
            "What is 15 plus 28?",
            "Convert 72 degrees Fahrenheit to Celsius.",
            "How much is a 20 percent tip on 38 dollars?",
            "What is the area of a rectangle that is 5 by 9 meters?",
            "Can you help me check this budget total?",
        ],
    ),
    (
        "sensor_status_verbal",
        [
            "What do you see in front of you?",
            "Is the room bright right now?",
            "Can you tell me if the door looks open?",
            "Do you detect any obstacles nearby?",
            "What is your battery level?",
            "Can you report the temperature if you know it?",
            "Is there a person near the table?",
            "Can you describe the objects on the desk?",
            "Do you hear any unusual noise?",
            "What is your current connection status?",
        ],
    ),
    (
        "troubleshooting_help",
        [
            "My laptop will not connect to Wi-Fi; what should I try?",
            "Why is my phone battery draining so fast?",
            "How do I reset a stuck application?",
            "Can you help me debug this error message?",
            "What should I check if the printer is offline?",
            "How can I make my computer run faster?",
            "Why is my microphone not working in calls?",
            "What does this warning light mean?",
            "Can you walk me through restarting the router?",
            "How do I recover a forgotten password safely?",
        ],
    ),
    (
        "safety_advice",
        [
            "What should I do if I smell gas?",
            "How can I clean up broken glass safely?",
            "What is the safest way to lift a heavy box?",
            "Should I touch a wet power outlet?",
            "How do I check whether food is spoiled?",
            "What should I do during a small kitchen fire?",
            "Can you explain basic first aid for a minor cut?",
            "How do I make this walkway less slippery?",
            "What are safe steps during an earthquake?",
            "How can I store cleaning chemicals safely?",
        ],
    ),
    (
        "opinions_reflection",
        [
            "Do you think routines are helpful?",
            "What makes a conversation feel respectful?",
            "Why do people like music?",
            "What is your view on lifelong learning?",
            "Can you compare remote work and office work?",
            "What are the pros and cons of adopting a pet?",
            "Why is patience important?",
            "What does fairness mean in everyday life?",
            "Can you help me think about whether to switch majors?",
            "What qualities make a good teammate?",
        ],
    ),
    (
        "clarification_confirmation",
        [
            "Did you understand my last question?",
            "Can you repeat that in simpler words?",
            "What did you mean by 'calibrate'?",
            "Can you summarize your answer?",
            "Are you saying I should start with the small task?",
            "Can you give me an example?",
            "Please clarify the second point.",
            "Could you answer more briefly?",
            "Can you restate that without technical terms?",
            "What assumptions are you making?",
        ],
    ),
    (
        "hypothetical_scenarios",
        [
            "What would happen if the power went out?",
            "If I had only ten minutes, what should I do first?",
            "How would you handle a noisy room?",
            "What if two people ask you questions at once?",
            "Suppose I cannot find my badge; what should I do?",
            "If the meeting is canceled, how should I use the time?",
            "What would you say to a child asking about robots?",
            "If you could not move, how would you still help?",
            "What should I do if the elevator is busy?",
            "How might a robot learn a new household routine?",
        ],
    ),
    (
        "action_capability_discussion",
        [
            "Can you describe how you would wave?",
            "What kinds of gestures can you perform?",
            "Tell me the difference between a nod and a bow.",
            "Explain how a robot plans a safe path.",
            "What movements are hardest for you?",
            "Can you list the dances you know without doing them?",
            "How would you point to an object if asked?",
            "What does it mean for a robot to keep balance?",
            "Can you talk about your walking speed?",
            "Describe your arm range of motion.",
        ],
    ),
    (
        "no_motion_instruction",
        [
            "Please answer with words only.",
            "Do not move; just tell me the answer.",
            "Stay where you are and explain the process.",
            "I only need a verbal response.",
            "Please describe it instead of demonstrating.",
            "Can you tell me the route without pointing?",
            "Answer quietly without any gesture.",
            "Use text only for this one.",
            "Please do not act it out.",
            "Give me the explanation while remaining still.",
        ],
    ),
    (
        "personalization_memory",
        [
            "What name did I ask you to call me?",
            "Can you remember that I prefer short answers?",
            "What did I say my favorite color was?",
            "Please note that I like direct feedback.",
            "Can you summarize my preferences so far?",
            "What have you learned about my morning routine?",
            "Do you remember my coffee order?",
            "Can you store that I prefer metric units?",
            "What should you keep in mind when helping me?",
            "Can you adapt your answers to my reading level?",
        ],
    ),
    (
        "social_etiquette",
        [
            "How should I greet a guest at the door?",
            "What is a polite way to interrupt someone?",
            "Can you help me apologize for being late?",
            "What should I say when someone gives me a gift?",
            "How do I introduce two coworkers?",
            "Can you draft a respectful disagreement?",
            "What is good etiquette during a video meeting?",
            "How can I thank a host after dinner?",
            "What should I say to welcome a new teammate?",
            "Can you help me respond to a compliment?",
        ],
    ),
    (
        "health_wellness_info",
        [
            "What are signs that I should drink more water?",
            "Can you suggest a gentle stretching routine in words?",
            "How much sleep do adults usually need?",
            "What are common ways to reduce eye strain?",
            "Can you explain why posture matters?",
            "What is a balanced breakfast idea?",
            "How do I start meditating?",
            "What should I ask a doctor about my cough?",
            "Can you describe breathing exercises without demonstrating?",
            "What are general tips for better sleep?",
        ],
    ),
    (
        "work_study_assistance",
        [
            "Can you help me outline this report?",
            "What questions should I ask in the interview?",
            "Summarize this meeting topic for me.",
            "How should I prepare for a coding test?",
            "Can you make a study quiz about biology?",
            "What is the best way to take notes?",
            "Help me write an email to my professor.",
            "Can you review my project plan conceptually?",
            "What should I include in a status update?",
            "How do I negotiate a deadline politely?",
        ],
    ),
    (
        "entertainment_conversation",
        [
            "Tell me a clean joke.",
            "Can we play a word guessing game verbally?",
            "Ask me a trivia question.",
            "What is a fun fact about space?",
            "Can you tell me a mystery riddle?",
            "Let's have a conversation about movies.",
            "What game can we play without moving?",
            "Tell me a story about a clever engineer.",
            "Can you invent a silly nickname for my backpack?",
            "What would be a good icebreaker question?",
        ],
    ),
]


MOTION_GROUPS = [
    (
        "greeting_gestures",
        [
            "Can you wave hello?",
            "Please give me a friendly wave.",
            "Greet the guest with a hand wave.",
            "Raise your hand to say hi.",
            "Can you do a small welcome gesture?",
            "Please bow politely to welcome them.",
            "Show me a cheerful greeting pose.",
            "Can you salute the team?",
            "Give me a thumbs-up greeting.",
            "Turn toward the visitor and wave.",
        ],
    ),
    (
        "yes_no_gestures",
        [
            "Can you nod if the answer is yes?",
            "Shake your head to say no.",
            "Use a thumbs up if you agree.",
            "Give me a thumbs down if that is wrong.",
            "Please answer yes with a head nod.",
            "Can you signal no without speaking?",
            "Show agreement with a simple gesture.",
            "Indicate uncertainty with a shrug.",
            "Tilt your head if you are not sure.",
            "Use your arm to signal approval.",
        ],
    ),
    (
        "dance_performance",
        [
            "Can you do a short dance?",
            "Dance for ten seconds.",
            "Show me your best robot dance.",
            "Can you perform a tiny victory dance?",
            "Please do a slow dance move.",
            "Move to the beat of this song.",
            "Give us a celebratory dance.",
            "Can you dance in place?",
            "Do a quick dance for the kids.",
            "Show a simple two-step dance.",
        ],
    ),
    (
        "exercise_stretch_demo",
        [
            "Show me a shoulder stretch.",
            "Can you demonstrate a squat?",
            "Do three jumping jacks.",
            "Please show a basic yoga pose.",
            "Stretch your arms overhead.",
            "Demonstrate how to warm up before running.",
            "Can you do a gentle side bend?",
            "Show me a balance pose.",
            "Perform a slow arm circle.",
            "Can you demonstrate a calf stretch?",
        ],
    ),
    (
        "pointing_reference",
        [
            "Point to the nearest chair.",
            "Can you point at the exit?",
            "Show me which button to press by pointing.",
            "Point toward the kitchen.",
            "Use your arm to indicate the red box.",
            "Can you gesture toward the person speaking?",
            "Point to where I should stand.",
            "Indicate the correct shelf with your hand.",
            "Show me the left door with a point.",
            "Can you point at the object you mean?",
        ],
    ),
    (
        "orientation_turning",
        [
            "Turn to face me.",
            "Rotate to your left.",
            "Please turn around.",
            "Face the window.",
            "Turn toward the sound.",
            "Can you look at the whiteboard?",
            "Pivot ninety degrees to the right.",
            "Orient yourself toward the charging dock.",
            "Please look away from the screen.",
            "Face the person who is speaking.",
        ],
    ),
    (
        "locomotion_basic",
        [
            "Move forward one meter.",
            "Step back a little.",
            "Please walk to the doorway.",
            "Come closer to me.",
            "Move to the center of the room.",
            "Back up until you are clear of the table.",
            "Can you follow me to the kitchen?",
            "Go to the charging station.",
            "Move aside so I can pass.",
            "Walk slowly beside me.",
        ],
    ),
    (
        "object_manipulation",
        [
            "Pick up the blue cup.",
            "Hand me the marker.",
            "Place the book on the table.",
            "Can you carry this box to the shelf?",
            "Push the cart forward.",
            "Open the drawer if you can.",
            "Press the green button.",
            "Close the cabinet door.",
            "Move the chair closer to the desk.",
            "Put the toy back in the bin.",
        ],
    ),
    (
        "social_interaction_touch",
        [
            "Give me a high five.",
            "Can you offer a fist bump?",
            "Shake my hand.",
            "Hold out your hand for the badge.",
            "Reach out so I can place the item in your hand.",
            "Can you tap my shoulder gently to get my attention?",
            "Offer your hand to the guest.",
            "Give the child a gentle high five.",
            "Extend your arm for a handshake.",
            "Can you do a safe elbow bump?",
        ],
    ),
    (
        "imitation_mimicry",
        [
            "Copy my pose.",
            "Mirror my arm movement.",
            "Can you imitate this dance step?",
            "Do what I do with my hands.",
            "Mimic my head tilt.",
            "Follow my clapping rhythm.",
            "Repeat the gesture I just made.",
            "Can you match my walking pace?",
            "Mirror my stretch.",
            "Copy the way I point.",
        ],
    ),
    (
        "demonstration_instructional",
        [
            "Show me how to lift a box safely.",
            "Demonstrate how to wave someone over.",
            "Can you act out opening a door?",
            "Show the motion for brushing teeth.",
            "Demonstrate how to tie an imaginary knot.",
            "Act out checking your pockets.",
            "Can you show how to scan a room?",
            "Demonstrate the correct hand position.",
            "Show me how to signal stop with your hand.",
            "Can you act out placing dishes on a shelf?",
        ],
    ),
    (
        "expressive_emotion_gesture",
        [
            "Show me a happy pose.",
            "Act surprised.",
            "Can you look confused?",
            "Show disappointment with your body language.",
            "Make a proud victory pose.",
            "Can you act sleepy?",
            "Show excitement without speaking.",
            "Pretend to be thinking deeply.",
            "Make a calm reassuring gesture.",
            "Show me a playful pose.",
        ],
    ),
    (
        "attention_and_gaze",
        [
            "Look at me while I speak.",
            "Make eye contact with the guest.",
            "Track my hand with your head.",
            "Watch the doorway.",
            "Follow the ball with your gaze.",
            "Look down at the floor marker.",
            "Can you scan the room from left to right?",
            "Keep your camera pointed at the table.",
            "Look at the person raising their hand.",
            "Turn your head toward the screen.",
        ],
    ),
    (
        "stop_stay_safety",
        [
            "Stop moving right now.",
            "Freeze in place.",
            "Stay still until I say go.",
            "Hold your current pose.",
            "Do not come any closer.",
            "Keep your arms down.",
            "Move away from the edge.",
            "Back away from the spill.",
            "Lower your arm slowly.",
            "Pause your movement.",
        ],
    ),
    (
        "delivery_fetch",
        [
            "Bring me the water bottle.",
            "Go get the mail from the table.",
            "Deliver this note to Sam.",
            "Carry the tray to the counter.",
            "Fetch the small towel.",
            "Take this cup to the sink.",
            "Can you bring the remote control?",
            "Move the package to the entryway.",
            "Please retrieve my glasses from the desk.",
            "Carry this folder to the office.",
        ],
    ),
    (
        "cleaning_tidying",
        [
            "Wipe the table.",
            "Pick up the papers from the floor.",
            "Put the dishes in the rack.",
            "Sweep this small area.",
            "Can you throw this wrapper in the bin?",
            "Straighten the chairs.",
            "Move these shoes against the wall.",
            "Clean the whiteboard with the eraser.",
            "Collect the cups from the meeting room.",
            "Tidy the cables under the desk.",
        ],
    ),
    (
        "navigation_guiding_by_motion",
        [
            "Lead me to the elevator.",
            "Guide me to the reception desk.",
            "Walk ahead and show me the way.",
            "Take me to the lab entrance.",
            "Can you escort the visitor to the lobby?",
            "Show me the route by moving there first.",
            "Guide the group to the cafeteria.",
            "Please lead us to the emergency exit.",
            "Walk me to the meeting room.",
            "Can you physically show me where the printer is?",
        ],
    ),
    (
        "games_play_motion",
        [
            "Play charades with me.",
            "Act out a profession for me to guess.",
            "Show a charades clue for a movie title.",
            "Do the motion for rock in rock-paper-scissors.",
            "Can you play Simon Says and copy the command?",
            "Make a pose for freeze dance.",
            "Act out rowing a boat for the guessing game.",
            "Pretend to throw an invisible ball.",
            "Clap twice for the rhythm game.",
            "Show a victory gesture when you win.",
        ],
    ),
    (
        "ceremony_presentation",
        [
            "Present the award with a bow.",
            "Unveil the sign by pulling the cover.",
            "Stand beside the speaker and gesture to them.",
            "Hold up the certificate for the photo.",
            "Raise the flag for the ceremony.",
            "Step onto the marker for the group picture.",
            "Hand the microphone to the next speaker.",
            "Point the audience toward the display.",
            "Give a closing bow.",
            "Pose next to the display for the announcement.",
        ],
    ),
    (
        "calibration_robot_control",
        [
            "Raise your left arm for calibration.",
            "Rotate your wrist slowly.",
            "Move your head from side to side.",
            "Open and close your gripper.",
            "Take one step forward for testing.",
            "Run your arm through its full range of motion.",
            "Stand on the floor marker.",
            "Turn in place for sensor calibration.",
            "Lift your right hand to shoulder height.",
            "Extend both arms straight ahead.",
        ],
    ),
    (
        "accessibility_assistance_motion",
        [
            "Open the door for me.",
            "Move the chair so I can sit.",
            "Press the elevator button.",
            "Hold the bag while I tie my shoe.",
            "Clear a path to the table.",
            "Bring the cane from the corner.",
            "Guide my hand to the rail.",
            "Pick up the dropped keys.",
            "Turn the page for me.",
            "Hold the light over the book.",
        ],
    ),
    (
        "kitchen_food_motion",
        [
            "Stir the soup gently.",
            "Pass me the salt.",
            "Put the plate on the counter.",
            "Open the refrigerator door.",
            "Pour water into the cup if you can.",
            "Move the pan off the burner.",
            "Place the spoon beside the bowl.",
            "Carry the groceries to the kitchen.",
            "Close the oven door.",
            "Hand me a clean napkin.",
        ],
    ),
    (
        "workshop_lab_motion",
        [
            "Press the emergency stop button.",
            "Place the sample on the tray.",
            "Hold the part steady.",
            "Turn the knob clockwise.",
            "Slide the component into the slot.",
            "Lift the lid carefully.",
            "Move the tool to the workbench.",
            "Point the camera at the circuit board.",
            "Pull the lever down.",
            "Set the container on the scale.",
        ],
    ),
    (
        "classroom_motion",
        [
            "Write the answer on the board.",
            "Point to the map.",
            "Hand out these worksheets.",
            "Raise your hand when you are ready.",
            "Stand next to the poster.",
            "Show the class how to fold the paper.",
            "Collect the pencils from the desks.",
            "Turn the page on the projector.",
            "Clap the syllables for this word.",
            "Demonstrate the science experiment motion.",
        ],
    ),
    (
        "ambiguous_but_action_required",
        [
            "Can you show me instead of telling me?",
            "Answer by moving your arm.",
            "Use a gesture to explain it.",
            "Can you make the response physical?",
            "Please demonstrate your answer.",
            "Show me what yes looks like.",
            "Act it out for me.",
            "Give me a nonverbal response.",
            "Use body language to answer.",
            "Can you respond with motion only?",
        ],
    ),
]


def build_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    for label, groups in (("chat", CHAT_GROUPS), ("motion_query", MOTION_GROUPS)):
        for scenario, utterances in groups:
            for utterance in utterances:
                examples.append(
                    {
                        "utterance": utterance,
                        "label": label,
                        "scenario": scenario,
                        "dataset_version": DATASET_VERSION,
                    }
                )

    random.Random(RANDOM_SEED).shuffle(examples)
    for index, example in enumerate(examples, start=1):
        example["id"] = f"seed-{index:04d}"

    return examples


def validate(examples: list[dict[str, str]]) -> None:
    if len(examples) != 500:
        raise ValueError(f"Expected 500 examples, found {len(examples)}.")

    utterances = [example["utterance"] for example in examples]
    duplicates = [text for text, count in Counter(utterances).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate utterances found: {duplicates[:5]}")

    label_counts = Counter(example["label"] for example in examples)
    expected_counts = {"chat": 250, "motion_query": 250}
    if dict(label_counts) != expected_counts:
        raise ValueError(f"Unexpected label distribution: {dict(label_counts)}")

    for example in examples:
        if not example["utterance"].isascii():
            raise ValueError(f"Non-ASCII utterance found: {example['utterance']}")


def write_jsonl(examples: list[dict[str, str]]) -> None:
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=True) + "\n")


def write_csv(examples: list[dict[str, str]]) -> None:
    fieldnames = ["id", "utterance", "label", "scenario", "dataset_version"]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    examples = build_examples()
    validate(examples)
    write_jsonl(examples)
    write_csv(examples)

    label_counts = Counter(example["label"] for example in examples)
    print(f"Wrote {len(examples)} examples to {JSONL_PATH}")
    print(f"Wrote {len(examples)} examples to {CSV_PATH}")
    print(f"Label distribution: {dict(label_counts)}")


if __name__ == "__main__":
    main()
