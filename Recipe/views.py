


# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# import google.generativeai as genai
# from django.conf import settings

# # Configure Google Generative AI
# genai.configure(api_key=settings.GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")

# def get_gemini_response(input_prompt, image_str):
#     if input_prompt:
#         response = model.generate_content([input_prompt])
#     else:
#         response = model.generate_content(image_str)
#     return response.text

# @csrf_exempt  # Disable CSRF for testing via Postman
# def ats_home(request):
#     if request.method == 'POST':
#         try:
#             # Parse JSON data from the request body
#             data = json.loads(request.body)
#             general_input = data.get('general_input', '')
#             image_str = data.get('image_str', '')
#             session_id=data.get('session_id')

#             if 'submit_recipe' in data:
#                 recipe_prompt = '''
#                 You are a Master chef who knows recipes from all over the world: Indian, Italian, American, Russian, Brazilian, etc. Now:
#                 Analyze the uploaded image of the recipe and provide detailed information as follows:
#                 you have to reply step by step as you are a assistant chef and you must provide the reply step by step teh user will ask you for next step tehn and the only you have to tell the next step 
#                 first step you shoukd tell:
#                 1. **List of Ingredients:** Identify and list all the items visible in the image.
#                 then user will provide a prompt again then and then only you should generate the next step
#                 2. **Recipe Instructions:** Generate a step-by-step recipe for preparing the dish shown in the image.
#                 3. **Preparation Method:** Describe the method for making the recipe, including any specific techniques or processes involved.
#                 4. **Precautions:** Mention any precautions or tips that should be considered while preparing the dish.
#                 Ensure the information is clear, comprehensive, and easy to follow.

#                 '''
#                 response = get_gemini_response(recipe_prompt, image_str)
#                 return JsonResponse({'response': response})

#             elif 'submit_general' in data:
#                 if general_input:
#                     response = get_gemini_response(general_input, "")
#                     return JsonResponse({'response': response})
#                 else:
#                     return JsonResponse({'error': "Please enter your query or input."})
        
#         except json.JSONDecodeError:
#             return JsonResponse({'error': "Invalid JSON data."})

#     # Return a method not allowed response for other HTTP methods
#     return JsonResponse({'error': "Method not allowed."}, status=405)

# import json
# import logging

# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# logger = logging.getLogger(__name__)
# import google.generativeai as genai

# def get_gemini_response(messages):
#     """
#     Gets a response from Gemini based on the provided conversation history.

#     Args:
#         messages (list): A list of message objects with varying formats.

#     Returns:
#         str: The generated text from Gemini.
#     """

#     def process_message(message):
#         """
#         Processes a message to ensure it has the required format and valid roles.

#         Args:
#             message (dict): A message dictionary with potential variations.

#         Returns:
#             dict: A dictionary with a "parts" key containing the message content.
#         """
#         valid_roles = {'user', 'model'}
#         if 'role' not in message or message['role'] not in valid_roles:
#             if message.get('role') == 'assistant':
#                 message['role'] = 'model'
#             else:
#                 raise ValueError(f"Invalid role: {message.get('role')}. Valid roles are: {valid_roles}")

#         if 'content' in message:
#             return {"role": message['role'], "parts": [message['content']]}
#         elif 'parts' in message:
#             return message  # Already has the expected format
#         else:
#             raise ValueError("Invalid message format")

#     # Process each message and create a consistent list
#     gemini_messages = [process_message(message) for message in messages]

#     # Create a Gemini model instance
#     model = genai.GenerativeModel('gemini-1.5-flash')

#     # Create a chat instance
#     chat = model.start_chat(history=gemini_messages)

#     # Send the last user message as a new message
#     last_user_message = process_message(messages[-1])['parts'][0]  # Extract content from processed message
#     response = chat.send_message(last_user_message)

#     return response.text

# @csrf_exempt
# def chat_with_gemini(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             logger.debug(f"Received request data: {data}")

#             # Initialize conversation if not in session
#             if 'conversation' not in request.session:
#                 request.session['conversation'] = []
#                 logger.debug("Initialized conversation in session")
            
#             messages = request.session['conversation']
#             logger.debug(f"Conversation history (roles): {[message['role'] for message in messages]}")

#             # Extract user message
#             user_message = data.get('prompt', '')
#             if not user_message:
#                 return JsonResponse({'error': 'Prompt is required'}, status=400)
#             logger.debug(f"User message: {user_message}")

#             # Append user message to conversation history
#             request.session['conversation'].append({'role': 'user', 'content': user_message})
#             messages = request.session['conversation']
#             logger.debug(f"Conversation history: {messages}")  # Print the conversation list

#             # Get Gemini response
#             gemini_reply = get_gemini_response(messages)
#             logger.debug(f"Gemini reply: {gemini_reply}")

#             # Append Gemini response to conversation history
#             request.session['conversation'].append({'role': 'model', 'content': gemini_reply})

#             # Save session changes
#             request.session.modified = True

#             # Return Gemini response to client
#             return JsonResponse({'reply': gemini_reply}, status=200)

#         except json.JSONDecodeError:
#             logger.error("Invalid JSON received")
#             return JsonResponse({'error': 'Invalid JSON'}, status=400)
#         except ValueError as e:
#             logger.error(f"ValueError: {str(e)}")
#             return JsonResponse({'error': str(e)}, status=400)
#         except Exception as e:
#             logger.exception(f"An error occurred: {str(e)}")
#             return JsonResponse({'error': 'Internal Server Error'}, status=500)
#     else:
#         return JsonResponse({'error': 'Invalid request method'}, status=405)

from django.conf import settings

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
def gemini(request):
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

            # Recipe prompt
            recipe_prompt = '''
                You are a Master chef who knows recipes from all over the world: Indian, Italian, American, Russian, Brazilian, etc. Now:
                you will assist the user,you will be provided with some igredients that user has you have to tell the user 3 recipes which he can make using selected ingrediets
                you are like a shef or cooking assistance you have to assist the user in making the recipe but kindly remeber you have to tell user when he asks means promt is provided
                you have to reply youa re able to know what prompt is provided to you so just dont tell whole recipe tell one by one step by step means first tell initial step ansd the user will ask somthing 
                then tell next step and aask user if any difficulty is there and once all recipe is completed tell the user recipe is done
                '''

            # Combine recipe prompt with user message
            combined_prompt = f"{recipe_prompt}\n\n{user_message}"

            # Append combined message to conversation history
            request.session['conversation'].append({'role': 'user', 'content': combined_prompt})
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

# ###recipe
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import logging
import requests
import base64
# from genai import GenerativeModel

logger = logging.getLogger(__name__)

@csrf_exempt
def image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.debug(f"Received request data: {data}")

            # Initialize conversation if not in session
            if 'conversation' not in request.session:
                request.session['conversation'] = []
                logger.debug("Initialized conversation in session")

            messages = request.session['conversation']
            logger.debug(f"Conversation history: {messages}")

            # Extract user message, image URL, and session ID
            user_message = data.get('prompt', '')
            image_url = data.get('image_url', '')
            session_id = data.get('session_id', '')

            if not user_message and not image_url:
                return JsonResponse({'error': 'Prompt or image_url is required'}, status=400)
            logger.debug(f"User message: {user_message}, Image URL: {image_url}")

            # Handle image upload and prompt
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            # Fetch and process the image
            image = None
            if image_url:
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()  # Check if the request was successful
                    image_data = response.content
                    # Encode image data to base64
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    image = {
                        'data': encoded_image,
                        'mime_type': response.headers['content-type'],
                    }
                except requests.RequestException as e:
                    logger.error(f"Error fetching image: {str(e)}")
                    return JsonResponse({'error': 'Error fetching image'}, status=400)

            # Prepare the content for the model
            parts = []
            if image:
                parts.append({'inline_data': image})
            if user_message:
                parts.append({'text': user_message})

            # Include conversation history in the model input
            conversation_parts = []
            for message in messages:
                if message['role'] == 'user':
                    conversation_parts.append({'text': message['content']})
                elif message['role'] == 'model':
                    conversation_parts.append({'text': message['content']})

            # Combine conversation parts with current parts
            conversation_parts.extend(parts)
            result = model.generate_content({'parts': conversation_parts})

            # Access the generated content correctly
            gemini_reply = result.text  # Adjusted to the correct attribute or method

            logger.debug(f"Gemini reply: {gemini_reply}")

            # Append user message and Gemini response to conversation history
            if user_message:
                request.session['conversation'].append({'role': 'user', 'content': user_message})
            if image:
                request.session['conversation'].append({'role': 'user', 'content': '[Image uploaded]'})
            request.session['conversation'].append({'role': 'model', 'content': gemini_reply})

            # Save session changes
            request.session.modified = True

            # Return Gemini response to client
            return JsonResponse({'response': gemini_reply}, status=200)

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

@csrf_exempt
def diet(request):
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

            # Recipe prompt
            recipe_prompt = '''
                you will be provided with some input word of common problems which sets a restriction to your diet and a word like weekly or daily ,
                so you have to plan a diet accordingly means if weekly present  plan for week and for daily plan daily and prepaer the reponse in well structured manner
                like generate a list of diet plan accordingly as you wish so as user can be claer what he/she needs what amount of calories,protiens carbohydrates fats excetra exetra it has mention that thing;
                and also tell the necessary precaution and time to follow 
                '''

            # Combine recipe prompt with user message
            combined_prompt = f"{recipe_prompt}\n\n{user_message}"

            # Append combined message to conversation history
            request.session['conversation'].append({'role': 'user', 'content': combined_prompt})
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



    