


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import google.generativeai as genai
from django.conf import settings

# Configure Google Generative AI
genai.configure(api_key=settings.GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input_prompt, image_str):
    if input_prompt:
        response = model.generate_content([input_prompt])
    else:
        response = model.generate_content(image_str)
    return response.text

@csrf_exempt  # Disable CSRF for testing via Postman
def ats_home(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)
            general_input = data.get('general_input', '')
            image_str = data.get('image_str', '')
            session_id=data.get('session_id')

            if 'submit_recipe' in data:
                recipe_prompt = '''
                You are a Master chef who knows recipes from all over the world: Indian, Italian, American, Russian, Brazilian, etc. Now:
                Analyze the uploaded image of the recipe and provide detailed information as follows:
                1. **List of Ingredients:** Identify and list all the items visible in the image.
                2. **Recipe Instructions:** Generate a step-by-step recipe for preparing the dish shown in the image.
                3. **Preparation Method:** Describe the method for making the recipe, including any specific techniques or processes involved.
                4. **Precautions:** Mention any precautions or tips that should be considered while preparing the dish.
                Ensure the information is clear, comprehensive, and easy to follow.
                '''
                response = get_gemini_response(recipe_prompt, image_str)
                return JsonResponse({'response': response})

            elif 'submit_general' in data:
                if general_input:
                    response = get_gemini_response(general_input, "")
                    return JsonResponse({'response': response})
                else:
                    return JsonResponse({'error': "Please enter your query or input."})
        
        except json.JSONDecodeError:
            return JsonResponse({'error': "Invalid JSON data."})

    # Return a method not allowed response for other HTTP methods
    return JsonResponse({'error': "Method not allowed."}, status=405)

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)
import google.generativeai as genai

def get_gemini_response(messages):
    """
    Gets a response from Gemini based on the provided conversation history.

    Args:
        messages (list): A list of message objects with varying formats.

    Returns:
        str: The generated text from Gemini.
    """

    def process_message(message):
        """
        Processes a message to ensure it has the required format and valid roles.

        Args:
            message (dict): A message dictionary with potential variations.

        Returns:
            dict: A dictionary with a "parts" key containing the message content.
        """
        valid_roles = {'user', 'model'}
        if 'role' not in message or message['role'] not in valid_roles:
            if message.get('role') == 'assistant':
                message['role'] = 'model'
            else:
                raise ValueError(f"Invalid role: {message.get('role')}. Valid roles are: {valid_roles}")

        if 'content' in message:
            return {"role": message['role'], "parts": [message['content']]}
        elif 'parts' in message:
            return message  # Already has the expected format
        else:
            raise ValueError("Invalid message format")

    # Process each message and create a consistent list
    gemini_messages = [process_message(message) for message in messages]

    # Create a Gemini model instance
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Create a chat instance
    chat = model.start_chat(history=gemini_messages)

    # Send the last user message as a new message
    last_user_message = process_message(messages[-1])['parts'][0]  # Extract content from processed message
    response = chat.send_message(last_user_message)

    return response.text

@csrf_exempt
def chat_with_gemini(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.debug(f"Received request data: {data}")

            # Initialize conversation if not in session
            if 'conversation' not in request.session:
                request.session['conversation'] = []
                logger.debug("Initialized conversation in session")
            
            messages = request.session['conversation']
            logger.debug(f"Conversation history (roles): {[message['role'] for message in messages]}")

            # Extract user message
            user_message = data.get('prompt', '')
            if not user_message:
                return JsonResponse({'error': 'Prompt is required'}, status=400)
            logger.debug(f"User message: {user_message}")

            # Append user message to conversation history
            request.session['conversation'].append({'role': 'user', 'content': user_message})
            messages = request.session['conversation']
            logger.debug(f"Conversation history: {messages}")  # Print the conversation list

            # Get Gemini response
            gemini_reply = get_gemini_response(messages)
            logger.debug(f"Gemini reply: {gemini_reply}")

            # Append Gemini response to conversation history
            request.session['conversation'].append({'role': 'model', 'content': gemini_reply})

            # Save session changes
            request.session.modified = True

            # Return Gemini response to client
            return JsonResponse({'reply': gemini_reply}, status=200)

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except ValueError as e:
            logger.error(f"ValueError: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
