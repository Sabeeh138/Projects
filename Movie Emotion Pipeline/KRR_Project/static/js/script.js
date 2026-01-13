document.addEventListener('DOMContentLoaded', function() {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');

    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    window.sendMessage = async function() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        userInput.value = '';
        
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot loading';
        loadingDiv.id = loadingId;
        loadingDiv.textContent = '🤔 Thinking...';
        chatArea.appendChild(loadingDiv);
        chatArea.scrollTop = chatArea.scrollHeight;

        try {
            const formData = new FormData();
            formData.append('message', text);

            const response = await fetch('/api/chat/', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log("Raw response data:", data); // Debug log
            
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();

            if (data.error) {
                console.log("Error in response:", data.error);
                appendMessage("Sorry, I encountered an error: " + data.error, 'bot');
            } else {
                console.log("Response type:", data.type); // Debug log
                console.log("Response text:", data.response); // Debug log
                
                // Handle different response types
                if (data.type === "conversation") {
                    console.log("Handling conversation response");
                    // Just conversation - no movie cards
                    appendMessage(data.response, 'bot', true);
                } else if (data.type === "movie_recommendation") {
                    console.log("Handling movie recommendation response");
                    // Movie recommendations with reasoning
                    let reply = data.response;
                    
                    // Add movie cards if there are movies
                    if (data.movies && data.movies.length > 0) {
                        console.log("Adding movie cards:", data.movies.length);
                        data.movies.forEach((movie, index) => {
                            // Generate reasoning for each movie based on emotion
                            let reasoning = generateMovieReasoning(movie, data.emotion);
                            
                            reply += `
                                <div class="recommendation-card">
                                    <h4>${movie.title} (${movie.year})</h4>
                                    <p><strong>Director:</strong> ${movie.director}</p>
                                    <p><strong>Genre:</strong> ${movie.genre}</p>
                                    <p><strong>Why this movie:</strong> ${reasoning}</p>
                                    <p><em>"${movie.review_snippet}"</em></p>
                                </div>
                            `;
                        });
                        
                        // Add a natural follow-up
                        const followUps = [
                            "What do you think? Any of these catch your interest?",
                            "Do these sound good to you?",
                            "How do these sound for your current mood?",
                            "Let me know what you think of these picks!"
                        ];
                        reply += `<br><p>${followUps[Math.floor(Math.random() * followUps.length)]}</p>`;
                    }
                    
                    appendMessage(reply, 'bot', true);
                } else {
                    // Fallback - this shouldn't happen with new system
                    console.log("Using fallback response format - this shouldn't happen!");
                    let reply = data.response || `I detected that you are feeling <strong>${data.emotion}</strong>.`;
                    
                    if (data.movies && data.movies.length > 0) {
                        reply += "<br><br>Here are some recommendations:";
                        data.movies.forEach(movie => {
                            reply += `
                                <div class="recommendation-card">
                                    <h4>${movie.title} (${movie.year})</h4>
                                    <p><strong>Director:</strong> ${movie.director}</p>
                                    <p><strong>Genre:</strong> ${movie.genre}</p>
                                    <p><em>"${movie.review_snippet}"</em></p>
                                </div>
                            `;
                        });
                    }
                    
                    appendMessage(reply, 'bot', true);
                }
            }

        } catch (error) {
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();
            appendMessage("Sorry, something went wrong with the server. Please try again!", 'bot');
            console.error(error);
        }
    }

    function generateMovieReasoning(movie, emotion) {
        // More thoughtful reasoning based on movie details and emotion
        let reasoning = "";
        
        // Analyze the movie's genre and how it relates to the emotion
        const genre = movie.genre ? movie.genre.toLowerCase() : "";
        
        if (emotion === "sadness") {
            if (genre.includes("comedy")) {
                reasoning = `This comedy can provide the emotional lift you need right now. Laughter has a way of breaking through sadness and giving you a fresh perspective`;
            } else if (genre.includes("drama")) {
                reasoning = `This drama will let you explore and process your emotions in a safe space. Sometimes we need to feel our feelings fully to move through them`;
            } else if (genre.includes("romance")) {
                reasoning = `This romantic story can remind you of the beauty and connection that exists in the world, even during difficult times`;
            } else {
                reasoning = `This film offers the kind of emotional journey that can be cathartic when you're feeling down`;
            }
        } else if (emotion === "joy") {
            if (genre.includes("comedy")) {
                reasoning = `This comedy will amplify your good mood and keep those positive vibes flowing. When you're already happy, great humor just makes everything better`;
            } else if (genre.includes("adventure")) {
                reasoning = `This adventure matches your upbeat energy perfectly. When you're feeling good, exciting stories can make you feel even more alive`;
            } else if (genre.includes("musical")) {
                reasoning = `This musical will complement your joyful mood beautifully. Music and happiness go hand in hand`;
            } else {
                reasoning = `This uplifting story will keep your positive momentum going strong`;
            }
        } else if (emotion === "anger") {
            if (genre.includes("action")) {
                reasoning = `This action film provides a healthy outlet for that intense energy you're feeling. Sometimes we need to see conflict resolved on screen`;
            } else if (genre.includes("thriller")) {
                reasoning = `This thriller can help channel your frustration into engagement with a gripping story`;
            } else if (genre.includes("comedy")) {
                reasoning = `This comedy might help you laugh off some of that anger. Humor can be surprisingly effective at diffusing frustration`;
            } else {
                reasoning = `This film can help you process those intense feelings in a constructive way`;
            }
        } else if (emotion === "fear") {
            if (genre.includes("comedy")) {
                reasoning = `This comedy can help ease your anxiety with some gentle laughs. Humor is one of the best antidotes to worry`;
            } else if (genre.includes("feel-good") || genre.includes("family")) {
                reasoning = `This comforting film will help calm your nerves and remind you that things can work out okay`;
            } else {
                reasoning = `This soothing story can help quiet your anxious thoughts and provide some peace of mind`;
            }
        } else {
            reasoning = `This well-crafted film matches your current emotional state and should provide exactly what you need right now`;
        }
        
        // Add director-specific insights
        if (movie.director) {
            const directorInsights = {
                "Christopher Nolan": "Nolan's masterful storytelling will keep your mind engaged and provide the perfect escape",
                "Quentin Tarantino": "Tarantino's unique style and sharp dialogue create an immersive experience",
                "Steven Spielberg": "Spielberg's emotional intelligence as a filmmaker makes this especially powerful",
                "Martin Scorsese": "Scorsese's deep understanding of human nature shines through in every scene",
                "Greta Gerwig": "Gerwig's authentic approach to storytelling creates genuine emotional connections"
            };
            
            if (directorInsights[movie.director]) {
                reasoning += `. Plus, ${directorInsights[movie.director]}`;
            }
        }
        
        // Add year-based context
        if (movie.year) {
            const year = parseInt(movie.year);
            if (year < 1980) {
                reasoning += `. This classic has stood the test of time because it speaks to universal human experiences`;
            } else if (year < 2000) {
                reasoning += `. This gem from the ${Math.floor(year/10)*10}s has a timeless quality that still resonates today`;
            } else if (year < 2010) {
                reasoning += `. This 2000s film combines modern filmmaking with proven storytelling`;
            } else {
                reasoning += `. This recent film brings contemporary perspectives to timeless themes`;
            }
        }
        
        return reasoning + ".";
    }

    function appendMessage(text, sender, isHtml = false) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        if (isHtml) {
            div.innerHTML = text;
        } else {
            div.textContent = text;
        }
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }
});
