from django.shortcuts import render
from django.http import JsonResponse
import emotion_pipeline
from .models import Movie
from django.views.decorators.csrf import csrf_exempt
import json
import os
import random
from datetime import datetime

class SmartMovieChatbot:
    def __init__(self):
        self.memory_file = "user_memory.json"
        self.load_memory()
        
        # Keywords that indicate movie recommendation requests
        self.movie_request_keywords = [
            "recommend", "suggestion", "suggest", "movie", "film", "watch", "see",
            "recommendation", "what should i", "looking for", "want to see",
            "need something", "find me", "show me", "give me", "help me find"
        ]
        
        # Conversational responses for different scenarios
        self.conversation_responses = {
            "greeting": [
                "Hello! I'm your movie companion. How are you feeling today?",
                "Hi there! I'm here to chat or help you find the perfect movie. What's on your mind?",
                "Hey! Great to see you. How can I help you today?",
                "Hello! I'm ready to chat or find you some amazing movies. What's up?"
            ],
            "general_chat": [
                "That's interesting! Tell me more about that.",
                "I understand. How does that make you feel?",
                "That sounds like quite an experience. What's going through your mind?",
                "I hear you. Is there anything specific you'd like to talk about?",
                "Thanks for sharing that with me. How are you processing all of this?"
            ],
            "emotion_acknowledgment": {
                "joy": [
                    "I can feel your positive energy! That's wonderful to hear.",
                    "Your happiness is contagious! I love that you're feeling so good.",
                    "That's fantastic! It sounds like you're in a really great place right now."
                ],
                "sadness": [
                    "I'm sorry you're going through a tough time. I'm here to listen.",
                    "That sounds really difficult. Thank you for sharing that with me.",
                    "I can hear that you're hurting. It's okay to feel sad sometimes."
                ],
                "anger": [
                    "That sounds really frustrating. I can understand why you'd feel that way.",
                    "It sounds like something really got under your skin. That's totally valid.",
                    "I can sense your frustration. Sometimes things just don't go the way we want."
                ],
                "fear": [
                    "That sounds scary. It takes courage to talk about what frightens us.",
                    "I understand that you're feeling anxious. That's completely normal.",
                    "Fear can be overwhelming. I'm here if you want to talk about it."
                ],
                "neutral": [
                    "You seem pretty balanced right now. That's a good place to be.",
                    "I'm getting a calm vibe from you. How are things going in general?",
                    "You sound pretty centered. What's been on your mind lately?"
                ]
            }
        }
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.user_memory = json.load(f)
            else:
                self.user_memory = {"conversations": [], "recommendations": []}
        except:
            self.user_memory = {"conversations": [], "recommendations": []}
    
    def save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.user_memory, f, indent=2)
        except:
            pass
    
    def is_movie_request(self, message):
        """Determine if the user is asking for movie recommendations"""
        message_lower = message.lower().strip()
        
        # Simple greetings and responses should NOT be movie requests
        simple_responses = [
            "hi", "hello", "hey", "sup", "what's up", "how are you", 
            "good", "great", "fine", "okay", "bad", "terrible", "sad",
            "i'm good", "i'm fine", "i'm okay", "i'm great", "i'm bad",
            "not good", "not great", "feeling good", "feeling bad"
        ]
        if message_lower in simple_responses:
            return False
        
        # Questions about previous recommendations should NOT trigger new recommendations
        if any(phrase in message_lower for phrase in ["why these", "why those", "why did you", "explain", "reasoning"]):
            return False
        
        # Explicit movie request phrases - these are STRONG indicators
        strong_movie_phrases = [
            "recommend me a movie", "recommend a movie", "suggest a movie", "suggest me a movie",
            "recommend me something", "suggest me something", "recommend something",
            "movie recommendation", "film recommendation", "what movie", "what film",
            "find me a movie", "find me a film", "show me movies", "give me movies",
            "want to watch", "looking to watch", "need a movie", "need a film",
            "movie suggestions", "film suggestions", "any movies", "any films",
            "movies recommended", "have movies recommended", "would like to have movies recommended"
        ]
        
        for phrase in strong_movie_phrases:
            if phrase in message_lower:
                return True
        
        # Check for "yes" responses that might be answering movie offer
        if message_lower in ["yes", "yeah", "sure", "ok", "okay", "yes please", "sure thing", "sounds good"]:
            return True
        
        # Check for movie/film mentions with action words
        if ("movie" in message_lower or "film" in message_lower) and \
           any(word in message_lower for word in ["recommend", "suggest", "find", "show", "give", "want", "need", "looking"]):
            return True
        
        # Check for explicit recommendation requests
        if any(word in message_lower for word in ["recommend", "suggest"]) and \
           any(word in message_lower for word in ["something", "anything", "me"]):
            return True
        
        return False
    
    def is_greeting(self, message):
        """Check if message is a simple greeting"""
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "what's up", "sup", "hii", "hiii", "hello there"]
        return message.lower().strip() in greetings
    
    def analyze_user_message(self, message, emotion):
        """Analyze the user's message to understand context and provide thoughtful responses"""
        message_lower = message.lower()
        
        # Extract key information from the message
        analysis = {
            "main_topics": [],
            "emotional_indicators": [],
            "context_clues": [],
            "intensity": "moderate",
            "needs_support": False,
            "specific_concerns": []
        }
        
        # Analyze emotional intensity
        if any(word in message_lower for word in ["really", "very", "extremely", "so", "super", "totally"]):
            analysis["intensity"] = "high"
        elif any(word in message_lower for word in ["a bit", "kinda", "somewhat", "little", "slightly"]):
            analysis["intensity"] = "mild"
        
        # Identify specific topics and concerns
        if any(word in message_lower for word in ["work", "job", "boss", "colleague", "office"]):
            analysis["main_topics"].append("work")
            analysis["context_clues"].append("work-related stress")
        
        if any(word in message_lower for word in ["school", "exam", "test", "homework", "study", "grade"]):
            analysis["main_topics"].append("education")
            analysis["context_clues"].append("academic pressure")
        
        if any(word in message_lower for word in ["relationship", "boyfriend", "girlfriend", "partner", "love", "breakup"]):
            analysis["main_topics"].append("relationships")
            analysis["context_clues"].append("relationship matters")
        
        if any(word in message_lower for word in ["family", "parents", "mom", "dad", "sister", "brother"]):
            analysis["main_topics"].append("family")
            analysis["context_clues"].append("family dynamics")
        
        if any(word in message_lower for word in ["money", "financial", "bills", "debt", "expensive"]):
            analysis["main_topics"].append("finances")
            analysis["context_clues"].append("financial concerns")
        
        if any(word in message_lower for word in ["health", "sick", "doctor", "hospital", "pain"]):
            analysis["main_topics"].append("health")
            analysis["context_clues"].append("health concerns")
        
        # Identify emotional indicators
        if any(word in message_lower for word in ["crying", "tears", "can't stop", "overwhelming"]):
            analysis["emotional_indicators"].append("intense sadness")
            analysis["needs_support"] = True
        
        if any(word in message_lower for word in ["angry", "furious", "pissed off", "rage"]):
            analysis["emotional_indicators"].append("anger")
        
        if any(word in message_lower for word in ["worried", "anxious", "panic", "scared"]):
            analysis["emotional_indicators"].append("anxiety")
            analysis["needs_support"] = True
        
        if any(word in message_lower for word in ["excited", "thrilled", "amazing", "fantastic"]):
            analysis["emotional_indicators"].append("high positive emotion")
        
        # Identify specific concerns or events
        if any(word in message_lower for word in ["happened", "today", "yesterday", "this morning"]):
            analysis["specific_concerns"].append("recent event")
        
        if any(word in message_lower for word in ["always", "never", "every day", "constantly"]):
            analysis["specific_concerns"].append("ongoing pattern")
        
        if any(word in message_lower for word in ["don't know", "confused", "lost", "unsure"]):
            analysis["specific_concerns"].append("uncertainty")
        
        return analysis
    
    def generate_thoughtful_response(self, message, emotion, analysis):
        """Generate a thoughtful, reasoned response based on analysis"""
        
        # Start with acknowledgment based on analysis
        response_parts = []
        
        # Acknowledge the specific situation
        if analysis["main_topics"]:
            topic = analysis["main_topics"][0]
            if topic == "work":
                response_parts.append("I can hear that work has been challenging for you.")
            elif topic == "relationships":
                response_parts.append("Relationship situations can be really complex and emotionally draining.")
            elif topic == "family":
                response_parts.append("Family dynamics can be tough to navigate sometimes.")
            elif topic == "education":
                response_parts.append("School stress is real - there's a lot of pressure to perform.")
            elif topic == "finances":
                response_parts.append("Financial worries can be incredibly stressful and overwhelming.")
            elif topic == "health":
                response_parts.append("Health concerns are always scary and can affect everything else.")
        
        # Address emotional intensity
        if analysis["intensity"] == "high":
            if emotion == "sadness":
                response_parts.append("It sounds like you're really going through it right now. That level of sadness is exhausting.")
            elif emotion == "anger":
                response_parts.append("I can feel the intensity of your frustration. That kind of anger is hard to carry.")
            elif emotion == "fear":
                response_parts.append("That level of anxiety must be overwhelming. It's hard to think clearly when you're that worried.")
        elif analysis["intensity"] == "mild":
            response_parts.append("Even though it might seem minor, these feelings are still valid and worth talking about.")
        
        # Address specific concerns
        if "recent event" in analysis["specific_concerns"]:
            response_parts.append("It sounds like something specific happened recently that's affecting you.")
        elif "ongoing pattern" in analysis["specific_concerns"]:
            response_parts.append("When something becomes a constant in your life, it can feel really overwhelming.")
        elif "uncertainty" in analysis["specific_concerns"]:
            response_parts.append("Not knowing what to do or how to feel can be one of the most frustrating experiences.")
        
        # Provide thoughtful follow-up questions
        if analysis["needs_support"]:
            follow_ups = [
                "Do you have people in your life you can talk to about this?",
                "How long have you been dealing with this?",
                "What usually helps you when you're feeling this way?",
                "Is there anything specific that triggered this today?"
            ]
        else:
            if analysis["main_topics"]:
                topic = analysis["main_topics"][0]
                if topic == "work":
                    follow_ups = [
                        "What's been the most stressful part about work lately?",
                        "Is this a new situation or something that's been building up?",
                        "How is this affecting other parts of your life?"
                    ]
                elif topic == "relationships":
                    follow_ups = [
                        "How long has this been going on?",
                        "Have you been able to talk to them about how you're feeling?",
                        "What's been the hardest part to deal with?"
                    ]
                else:
                    follow_ups = [
                        "What's been weighing on you the most about this?",
                        "How has this been affecting your daily life?",
                        "What would help you feel better about this situation?"
                    ]
            else:
                follow_ups = [
                    "What's been going through your mind about this?",
                    "How has this been affecting you?",
                    "What do you think would help right now?"
                ]
        
        # Add a thoughtful follow-up question
        response_parts.append(follow_ups[0])  # Use the first relevant follow-up
        
        # Sometimes offer movie suggestion based on analysis
        if analysis["needs_support"] or analysis["intensity"] == "high":
            if random.random() < 0.4:  # 40% chance when they need support
                response_parts.append("You know what, sometimes a good movie can provide a helpful distraction. Would you like me to suggest something that might help?")
        
        return " ".join(response_parts)

    def generate_conversation_response(self, message, emotion):
        """Generate a conversational response with proper reasoning"""
        
        message_lower = message.lower().strip()
        
        # Handle greetings naturally
        if self.is_greeting(message):
            greetings = [
                "Hey! How's your day going?",
                "Hi there! What's up?",
                "Hello! How are you feeling today?",
                "Hey! Good to see you. What's on your mind?"
            ]
            return random.choice(greetings)
        
        # Handle questions about movie recommendations
        if any(phrase in message_lower for phrase in ["why these", "why those", "why did you", "explain these", "reasoning"]):
            explanations = [
                "Great question! I chose those movies based on your emotional state and what typically resonates with people feeling similar emotions. Each film has elements that can either help you process your feelings or provide the right kind of emotional experience you might need right now.",
                "I selected those based on how they match your current mood and emotional needs. The genres, themes, and emotional tones of those films are specifically chosen to either complement or help shift your emotional state in a positive way.",
                "Those recommendations came from analyzing your emotional state and matching it with films that have proven to resonate with people in similar situations. Each movie offers something different - whether it's emotional catharsis, distraction, or inspiration.",
                "I picked those because they align with your emotional profile right now. The directors, themes, and storytelling styles in those films are particularly effective for people experiencing similar feelings to yours."
            ]
            return random.choice(explanations)
        
        # Handle simple responses
        if message_lower in ["good", "fine", "okay", "alright", "not bad"]:
            responses = [
                "That's great to hear! What's been going well for you?",
                "Glad you're doing well! Anything exciting happening?",
                "Nice! What's been keeping you in good spirits?",
                "That's good! Tell me more about your day."
            ]
            return random.choice(responses)
        
        if message_lower in ["bad", "not good", "terrible", "awful", "horrible"]:
            responses = [
                "I'm sorry to hear that. What's been bothering you?",
                "That sounds tough. Do you want to talk about what's going on?",
                "I'm here to listen. What's been making things difficult?",
                "That doesn't sound fun. What's been weighing on you?"
            ]
            return random.choice(responses)
        
        # Handle "how are you" questions
        if any(phrase in message_lower for phrase in ["how are you", "how's it going", "what's up"]):
            responses = [
                "I'm doing great, thanks for asking! I'm here and ready to chat or help you find some awesome movies. How are you doing?",
                "I'm fantastic! I love talking with people about movies and life. How's your day been?",
                "I'm doing well! Always excited to meet someone new. What's been on your mind lately?",
                "I'm great, thanks! I'm here whenever you need to chat or want movie suggestions. How are you feeling today?"
            ]
            return random.choice(responses)
        
        # Analyze the message for better responses
        analysis = self.analyze_user_message(message, emotion)
        
        # Generate contextual responses based on emotion and content
        if emotion == "joy":
            responses = [
                "I love your positive energy! What's got you feeling so good today?",
                "That's awesome! It sounds like you're having a great time. Tell me more!",
                "Your happiness is contagious! What's been the highlight of your day?",
                "That's wonderful to hear! What's been making you feel so upbeat?"
            ]
            return random.choice(responses)
        
        elif emotion == "sadness":
            responses = [
                "I can hear that you're going through a tough time. I'm here to listen if you want to share more.",
                "That sounds really difficult. Sometimes it helps to talk about what's bothering us.",
                "I'm sorry you're feeling down. What's been on your mind?",
                "It takes courage to share when we're not feeling our best. What's been weighing on you?"
            ]
            return random.choice(responses)
        
        elif emotion == "anger":
            responses = [
                "That sounds really frustrating! What's been getting under your skin?",
                "I can sense your frustration. Sometimes it helps to vent about what's bothering us.",
                "That must be really annoying. What happened that's got you feeling this way?",
                "Sounds like something really got to you. Want to tell me what's going on?"
            ]
            return random.choice(responses)
        
        elif emotion == "fear":
            responses = [
                "That sounds stressful. What's been making you feel anxious?",
                "I can understand feeling worried about things. What's been on your mind?",
                "That must be nerve-wracking. Do you want to talk about what's concerning you?",
                "It's normal to feel anxious sometimes. What's been causing you stress?"
            ]
            return random.choice(responses)
        
        # Default conversational responses
        general_responses = [
            "That's interesting! Tell me more about that.",
            "I'd love to hear more about what you're thinking.",
            "That sounds like there's a lot on your mind. What's the main thing bothering you?",
            "Thanks for sharing that with me. How are you feeling about everything?",
            "I'm listening. What else is going through your head?",
            "That's a lot to process. What's been the most challenging part?",
            "I appreciate you opening up. What would help you feel better about this situation?"
        ]
        
        return random.choice(general_responses)
    
    def generate_movie_recommendation_response(self, message, emotion, movies):
        """Generate a response with movie recommendations and detailed reasoning"""
        
        # Analyze the user's request for better recommendations
        message_lower = message.lower()
        analysis = self.analyze_user_message(message, emotion)
        
        response_parts = []
        
        # Provide thoughtful analysis of their emotional state
        if emotion == "sadness":
            if analysis["intensity"] == "high":
                response_parts.append("I can tell you're really struggling right now. When we're deeply sad, movies can either help us process those feelings or provide gentle comfort.")
            else:
                response_parts.append("I sense you're feeling down. Let me find movies that can either help you work through these emotions or lift your spirits.")
        elif emotion == "joy":
            response_parts.append("I love that you're feeling positive! Let me find movies that will complement and enhance that good energy you have right now.")
        elif emotion == "anger":
            if "work" in analysis["main_topics"]:
                response_parts.append("Work frustration is tough. I'll find movies that either let you channel that energy constructively or help you decompress.")
            else:
                response_parts.append("I can feel your frustration. Movies can be great for either venting that energy or helping you cool down.")
        elif emotion == "fear":
            response_parts.append("Anxiety can be overwhelming. I'll recommend comforting movies that can help calm your mind and provide some peace.")
        
        # Explain the reasoning behind the selection process
        if analysis["main_topics"]:
            topic = analysis["main_topics"][0]
            if topic == "work":
                response_parts.append("Since you mentioned work stress, I'm focusing on movies that either provide escapism or help you process workplace frustrations.")
            elif topic == "relationships":
                response_parts.append("Given the relationship context, I'm selecting films that explore human connections in meaningful ways.")
            elif topic == "family":
                response_parts.append("With family on your mind, I'm choosing movies that deal with family dynamics and personal growth.")
        
        # Add the movie selection intro
        if movies:
            response_parts.append("Here are my carefully chosen recommendations:")
        else:
            response_parts.append("I'm having trouble finding the perfect matches right now. Could you tell me more about what kind of story or genre appeals to you when you're feeling this way?")
        
        return "\n\n".join(response_parts)
    
    def explain_movie_choice_naturally(self, movie, emotion):
        """Explain movie choice in natural language"""
        explanations = []
        
        # Emotion-based reasoning
        if emotion == "sadness":
            if movie.genre and "comedy" in movie.genre.lower():
                explanations.append("This comedy should help lift your spirits")
            elif movie.genre and "drama" in movie.genre.lower():
                explanations.append("This drama will let you feel your emotions in a cathartic way")
            else:
                explanations.append("This should help comfort you during a tough time")
        
        elif emotion == "joy":
            explanations.append("This will keep your good mood going strong")
        
        elif emotion == "anger":
            if movie.genre and "action" in movie.genre.lower():
                explanations.append("This action film will help you channel that energy")
            else:
                explanations.append("This should help you blow off some steam")
        
        elif emotion == "fear":
            explanations.append("This comforting film will help you relax")
        
        else:
            explanations.append("This matches your current vibe perfectly")
        
        # Add specific movie details naturally
        if movie.director:
            famous_directors = {
                "Christopher Nolan": "Nolan's brilliant storytelling",
                "Quentin Tarantino": "Tarantino's signature style",
                "Steven Spielberg": "Spielberg's masterful direction"
            }
            if movie.director in famous_directors:
                explanations.append(f"plus {famous_directors[movie.director]} makes it even better")
        
        return ", and ".join(explanations) + "."
    
    def explain_movie_choice(self, movie, emotion):
        """Explain why a specific movie was chosen"""
        explanations = []
        
        # Emotion-based reasoning
        emotion_reasoning = {
            "joy": f"This {movie.genre.lower() if movie.genre else 'movie'} will amplify your positive energy",
            "sadness": f"This {movie.genre.lower() if movie.genre else 'movie'} can either help you process your emotions or lift your spirits",
            "anger": f"This {movie.genre.lower() if movie.genre else 'movie'} can help channel your intense feelings",
            "fear": f"This comforting {movie.genre.lower() if movie.genre else 'movie'} will help you relax",
            "surprise": f"This {movie.genre.lower() if movie.genre else 'movie'} has the excitement you're looking for",
            "disgust": f"This uplifting {movie.genre.lower() if movie.genre else 'movie'} will cleanse your palate",
            "neutral": f"This well-crafted {movie.genre.lower() if movie.genre else 'movie'} is perfect for your balanced mood"
        }
        
        explanations.append(emotion_reasoning.get(emotion, emotion_reasoning["neutral"]))
        
        # Add specific details about the movie
        if movie.release_year and movie.release_year.isdigit():
            year = int(movie.release_year)
            if year < 1980:
                explanations.append("this classic has timeless appeal")
            elif year < 2000:
                explanations.append("this retro gem has nostalgic charm")
            elif year < 2010:
                explanations.append("this modern film has proven staying power")
            else:
                explanations.append("this recent film has contemporary relevance")
        
        return ", and ".join(explanations)

# Global chatbot instance
smart_chatbot = SmartMovieChatbot()

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        message = request.POST.get("message", "")
        if not message:
            return JsonResponse({"error": "No message provided"})
        
        try:
            # Analyze emotion - focus on the CURRENT message only
            emotion, tokens, scores = emotion_pipeline.classify(message)
            base_emotion = emotion.replace("_mixed", "")
            
            # Override emotion detection for clear emotional expressions
            message_lower = message.lower()
            if any(word in message_lower for word in ["sad", "down", "depressed", "upset", "blue", "feeling sad"]):
                base_emotion = "sadness"
            elif any(word in message_lower for word in ["happy", "great", "awesome", "fantastic", "feeling good", "feeling great"]):
                base_emotion = "joy"
            elif any(word in message_lower for word in ["angry", "mad", "frustrated", "pissed", "annoyed"]):
                base_emotion = "anger"
            elif any(word in message_lower for word in ["scared", "afraid", "worried", "anxious", "nervous"]):
                base_emotion = "fear"
            
            # Determine if this is a movie request or general conversation
            is_movie_request = smart_chatbot.is_movie_request(message)
            
            print(f"DEBUG: Message: '{message}', Is movie request: {is_movie_request}, Emotion: {base_emotion}")
            
            if is_movie_request:
                # Handle movie recommendation request
                movies = Movie.objects.filter(predicted_emotion=base_emotion)
                
                if not movies.exists():
                    # Fallback to similar emotions
                    emotion_fallbacks = {
                        "joy": ["surprise", "neutral"],
                        "sadness": ["fear", "neutral"],
                        "anger": ["disgust", "neutral"],
                        "fear": ["sadness", "neutral"],
                        "surprise": ["joy", "neutral"],
                        "disgust": ["anger", "neutral"],
                        "neutral": ["joy", "surprise"]
                    }
                    
                    fallback_emotions = emotion_fallbacks.get(base_emotion, ["neutral"])
                    for fallback in fallback_emotions:
                        movies = Movie.objects.filter(predicted_emotion=fallback)
                        if movies.exists():
                            break
                
                # Score and rank movies
                movie_list = []
                for m in movies:
                    score = 0.0
                    if m.emotion_scores:
                        parts = m.emotion_scores.split('|')
                        for p in parts:
                            if ':' in p:
                                k, v = p.split(':')
                                if k.strip() == base_emotion:
                                    try:
                                        score = float(v)
                                    except:
                                        pass
                                    break
                    movie_list.append((m, score))
                
                movie_list.sort(key=lambda x: x[1], reverse=True)
                
                # Get top 3 recommendations with variety
                recommendations = []
                used_genres = set()
                used_directors = set()
                
                for movie, score in movie_list:
                    if len(recommendations) >= 3:
                        break
                    
                    # Prefer variety
                    if (movie.genre not in used_genres or len(recommendations) == 0) and \
                       (movie.director not in used_directors or len(recommendations) < 2):
                        recommendations.append(movie)
                        used_genres.add(movie.genre)
                        used_directors.add(movie.director)
                
                # Fill remaining slots if needed
                while len(recommendations) < 3 and len(movie_list) > len(recommendations):
                    for movie, score in movie_list:
                        if movie not in recommendations:
                            recommendations.append(movie)
                            break
                
                # Generate movie recommendation response
                bot_response = smart_chatbot.generate_movie_recommendation_response(
                    message, base_emotion, recommendations
                )
                
                # Save to memory
                smart_chatbot.user_memory.setdefault("recommendations", []).append({
                    "message": message,
                    "emotion": base_emotion,
                    "movies": [m.title for m in recommendations],
                    "timestamp": datetime.now().isoformat()
                })
                smart_chatbot.save_memory()
                
                response_data = {
                    "type": "movie_recommendation",
                    "emotion": base_emotion,
                    "response": bot_response,
                    "movies": [
                        {
                            "title": m.title,
                            "director": m.director,
                            "year": m.release_year,
                            "genre": m.genre,
                            "review_snippet": m.review[:100] + "..." if m.review else ""
                        } for m in recommendations
                    ]
                }
                
            else:
                # Handle general conversation
                bot_response = smart_chatbot.generate_conversation_response(message, base_emotion)
                
                # Save to memory
                smart_chatbot.user_memory.setdefault("conversations", []).append({
                    "message": message,
                    "emotion": base_emotion,
                    "response": bot_response,
                    "timestamp": datetime.now().isoformat()
                })
                smart_chatbot.save_memory()
                
                response_data = {
                    "type": "conversation",
                    "emotion": base_emotion,
                    "response": bot_response,
                    "movies": []
                }
            
            print(f"DEBUG: Response type: {response_data['type']}")
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"DEBUG: Error: {str(e)}")
            return JsonResponse({
                "error": f"Processing error: {str(e)}",
                "debug": "Check server logs for details"
            })
        
    return JsonResponse({"error": "Invalid request method"})

def default(request):
    return render(request, 'default.html')

def chat(request):
    return render(request, 'chat.html')

def form(request):
    return render(request, 'form.html')
@csrf_exempt
def form_recommendation_api(request):
    if request.method == "POST":
        try:
            # Get form data
            primary_emotion = request.POST.get("primary_emotion", "")
            intensity = request.POST.get("intensity", "5")
            reason = request.POST.get("reason", "")
            genres = request.POST.getlist("genres")
            mood_goal = request.POST.get("mood_goal", "")
            context = request.POST.get("context", "")
            
            if not primary_emotion:
                return JsonResponse({"error": "Primary emotion is required"})
            
            # Create analysis text from form data
            analysis_parts = [f"You're feeling {primary_emotion}"]
            
            if intensity:
                intensity_level = "mild" if int(intensity) <= 3 else "moderate" if int(intensity) <= 7 else "very intense"
                analysis_parts.append(f"with {intensity_level} intensity")
            
            if reason:
                reason_map = {
                    "work": "due to work/career stress",
                    "relationships": "due to relationship issues", 
                    "family": "due to family matters",
                    "health": "due to health concerns",
                    "finances": "due to financial worries",
                    "school": "due to school/studies pressure",
                    "achievement": "due to a recent achievement",
                    "loss": "due to loss or disappointment",
                    "other": "due to other personal reasons"
                }
                analysis_parts.append(reason_map.get(reason, f"due to {reason}"))
            
            if mood_goal:
                goal_map = {
                    "enhance": "You want a movie to enhance your current mood",
                    "change": "You want a movie to change your mood completely",
                    "comfort": "You want a movie to provide comfort and relaxation",
                    "distract": "You want a movie to distract you from your problems",
                    "inspire": "You want a movie to inspire and motivate you",
                    "cathartic": "You want a movie to help you process your emotions"
                }
                analysis_parts.append(goal_map.get(mood_goal, ""))
            
            if context:
                analysis_parts.append(f"Additional context: {context}")
            
            analysis = ". ".join(filter(None, analysis_parts)) + "."
            
            # Get movie recommendations based on emotion
            base_emotion = primary_emotion.replace("_mixed", "")
            movies = Movie.objects.filter(predicted_emotion=base_emotion)
            
            if not movies.exists():
                # Fallback to similar emotions
                emotion_fallbacks = {
                    "joy": ["surprise", "neutral"],
                    "sadness": ["fear", "neutral"],
                    "anger": ["disgust", "neutral"],
                    "fear": ["sadness", "neutral"],
                    "surprise": ["joy", "neutral"],
                    "disgust": ["anger", "neutral"],
                    "neutral": ["joy", "surprise"]
                }
                
                fallback_emotions = emotion_fallbacks.get(base_emotion, ["neutral"])
                for fallback in fallback_emotions:
                    movies = Movie.objects.filter(predicted_emotion=fallback)
                    if movies.exists():
                        break
            
            # Filter by genres if specified
            if genres:
                genre_filter = movies.none()
                for genre in genres:
                    genre_filter = genre_filter | movies.filter(genre__icontains=genre)
                if genre_filter.exists():
                    movies = genre_filter
            
            # Score and rank movies
            movie_list = []
            for m in movies:
                score = 0.0
                if m.emotion_scores:
                    parts = m.emotion_scores.split('|')
                    for p in parts:
                        if ':' in p:
                            k, v = p.split(':')
                            if k.strip() == base_emotion:
                                try:
                                    score = float(v)
                                except:
                                    pass
                                break
                movie_list.append((m, score))
            
            movie_list.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 3 recommendations with variety
            recommendations = []
            used_genres = set()
            used_directors = set()
            
            for movie, score in movie_list:
                if len(recommendations) >= 3:
                    break
                
                # Prefer variety
                if (movie.genre not in used_genres or len(recommendations) == 0) and \
                   (movie.director not in used_directors or len(recommendations) < 2):
                    recommendations.append(movie)
                    used_genres.add(movie.genre)
                    used_directors.add(movie.director)
            
            # Fill remaining slots if needed
            while len(recommendations) < 3 and len(movie_list) > len(recommendations):
                for movie, score in movie_list:
                    if movie not in recommendations:
                        recommendations.append(movie)
                        break
            
            # Generate reasoning for each movie
            def generate_reasoning(movie, emotion, mood_goal):
                reasoning_parts = []
                
                if mood_goal == "enhance" and emotion == "joy":
                    reasoning_parts.append("This uplifting film will amplify your positive mood")
                elif mood_goal == "change" and emotion == "sadness":
                    reasoning_parts.append("This engaging movie will help shift your perspective")
                elif mood_goal == "comfort":
                    reasoning_parts.append("This comforting film provides the emotional warmth you're seeking")
                elif mood_goal == "distract":
                    reasoning_parts.append("This captivating movie will take your mind off your troubles")
                elif mood_goal == "inspire":
                    reasoning_parts.append("This inspiring story will motivate and uplift you")
                elif mood_goal == "cathartic":
                    reasoning_parts.append("This emotionally rich film will help you process your feelings")
                else:
                    reasoning_parts.append(f"This {movie.genre.lower() if movie.genre else 'film'} matches your current emotional state")
                
                if movie.genre:
                    if "comedy" in movie.genre.lower() and emotion == "sadness":
                        reasoning_parts.append("and the humor will provide a gentle emotional lift")
                    elif "drama" in movie.genre.lower():
                        reasoning_parts.append("and offers meaningful emotional depth")
                    elif "action" in movie.genre.lower() and emotion == "anger":
                        reasoning_parts.append("and provides a healthy outlet for intense emotions")
                
                return ". ".join(reasoning_parts) + "."
            
            response_data = {
                "emotion": base_emotion,
                "analysis": analysis,
                "movies": [
                    {
                        "title": m.title,
                        "director": m.director,
                        "year": m.release_year,
                        "genre": m.genre,
                        "review_snippet": m.review[:100] + "..." if m.review else "",
                        "reasoning": generate_reasoning(m, base_emotion, mood_goal)
                    } for m in recommendations
                ]
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({
                "error": f"Processing error: {str(e)}",
                "debug": "Check server logs for details"
            })
        
    return JsonResponse({"error": "Invalid request method"})