


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
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import logging
import requests
import base64
# from genai import GenerativeModel
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import logging
import requests
import base64
# from genai import GenerativeModel

logger = logging.getLogger(__name__)
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
import requests
import base64
# from genai import GenerativeModel
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
import requests
import base64
# from genai import GenerativeModel

logger = logging.getLogger(__name__)
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
import requests
import base64
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
import requests
import base64




from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.sessions.backends.db import SessionStore
import json
import logging
import requests
import base64

logger = logging.getLogger(__name__)

# Define a predefined prompt for crop disease diagnosis
PREDEFINED_PROMPT = ''' You are a master chef and you are been proivded with some data anaylze it and genereate contemt accordingly '''

@csrf_exempt
def image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.debug(f"Received request data: {data}")

            # Extract session ID from the request
            session_id = data.get('session_id', '')

            if not session_id:
                return JsonResponse({'error': 'Session ID is required'}, status=400)

            # Use Django's session framework with the provided session ID
            session = SessionStore(session_key=session_id)

            # Check if the session exists, if not, create it
            if not session.exists(session.session_key):
                session.create()
                session_id = session.session_key
                logger.debug(f"Created new session with session_id: {session_id}")
            else:
                logger.debug(f"Using existing session with session_id: {session_id}")

            # Initialize conversation if not in session
            if 'conversation' not in session:
                session['conversation'] = []
                logger.debug(f"Initialized conversation for session key: {session.session_key}")

            messages = session['conversation']
            logger.debug(f"Conversation history for session {session.session_key}: {messages}")

            # Extract image URL
            image_url = data.get('image_url', '')
            user_message = data.get('prompt', '')

           
            if not user_message and not image_url:
                return JsonResponse({'error': 'Prompt or image_url is required'}, status=400)
            logger.debug(f"User message: {user_message}, Image URL: {image_url}")

            # Handle image upload
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
            # Include the predefined prompt
            parts.append({'text': PREDEFINED_PROMPT})

            # Include conversation history in the model input
            conversation_parts = [{'text': message['content']} for message in messages]

            # Combine conversation parts with current parts
            conversation_parts.extend(parts)
            result = model.generate_content({'parts': conversation_parts})

            # Access the generated content correctly
            gemini_reply = result.text  # Adjusted to the correct attribute or method


            # Generate content with the Gemini model
            result = model.generate_content({'parts': conversation_parts})

# Assuming result is a response object with a 'text' attribute:
         
            logger.debug(f"Gemini reply: {gemini_reply}")




            # Append the Gemini response to conversation history
            session['conversation'].append({'role': 'model', 'content': gemini_reply})

            # Save session changes
            session.save()

            # Return Gemini response to client
            return JsonResponse({'response': gemini_reply, 'session_id': session_id}, status=200)

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



from django.shortcuts import render

def index(request):
    return render(request, 'index.html')







import os
import json
import requests
from io import BytesIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
os.environ['GENAI_API_KEY'] = 'AIzaSyAb_szzFcil2GC2UJHq_ooE6bb-Z9fkA3o'  # Replace with your actual API key

# Define your list of PDF URLs here
PDF_URLS = [
    
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/ALAHABAD%20High%20Court.pdf?alt=media&token=167623fa-bd50-451a-999c-937403109289",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/ANDHRA.pdf?alt=media&token=cae63b52-1a58-403d-a082-ed31cf0f5f05",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/BOMBAY.pdf?alt=media&token=67970fe5-10ff-46e2-954d-6a07b9d30da0",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/FAST%20Track%20Special%20Court.pdf?alt=media&token=df061683-e8fc-474e-a1da-e9ccfb2ca326",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/High%20Court%20Vacancy.pdf?alt=media&token=5e24ffbd-ccfe-4160-b875-a4893974422e",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/Higher%20Higher%20Judges.pdf?alt=media&token=d34cf137-0bb9-4b2a-bf3e-9c9dfcce3db9",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/JUDGES.pdf?alt=media&token=bd26d405-c58d-41e0-b6b6-2155d542dc32",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/Supreme%20Court%20Vacancy.pdf?alt=media&token=60926839-6614-43ed-9a6d-c5483d7d884d",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/Tele%20Law.pdf?alt=media&token=d71883a9-5b77-4fae-8723-55ba29c97e1e",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/Vacancy.pdf?alt=media&token=49f7d48b-c258-44d4-b6fd-f5ba447576f7",
             "https://firebasestorage.googleapis.com/v0/b/dojweb-de6e2.appspot.com/o/alldataofdojinpdf.pdf?alt=media&token=272e4f0f-35ce-4bac-afff-e9c86f7f43dd"
]

# Function to get text from a single PDF URL
def get_pdf_text_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF from URL {pdf_url}: {str(e)}")
        return None

# Function to get text from all PDF URLs
def get_text_from_all_pdfs():
    all_text = ""
    for url in PDF_URLS:
        pdf_text = get_pdf_text_from_url(url)
        if pdf_text:
            all_text += pdf_text + "\n"  # Separate each PDF's text with a newline
    return all_text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    You are provided with data from the Department of Justice in India. Try to answer the input question, including any relevant information.
    \n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@csrf_exempt
def process_request(request):
    if request.method == 'POST':
        try:
            input_json = json.loads(request.body.decode('utf-8'))
            prompt = input_json.get("prompt", "")

            if not prompt:
                return JsonResponse({"error": "Prompt is missing in the input."}, status=400)

            # Extract text from all the PDFs
            raw_text = get_text_from_all_pdfs()
            if not raw_text:
                return JsonResponse({"error": "Failed to extract text from PDFs."}, status=500)

            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Create vector store
            create_vector_store(text_chunks)

            # Load vector store and process the prompt
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(prompt)

            # Generate response using the conversational chain
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)

            return JsonResponse({"response": response["output_text"]}, status=200)
        
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Invalid request method."}, status=405)












# views.py

# from django.shortcuts import render
# from django.http import JsonResponse
# import google.generativeai as genai



# from django.shortcuts import render
# from django.http import JsonResponse
# import google.generativeai as genai
# import os


# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model=genai.GenerativeModel("gemini-1.5-flash")  

# # Define the response function
# def gemini_response(prompt):
#     if prompt:
#         response = model.generate_content(prompt)
#         if response:
#             return response.text
#         else:
#             return "No response received from the model."
#     else:
#         return "No input prompt provided."

# # Define the Django view function
# def gemini_llm_view(request):
#     if request.method == "POST":
#         # Get the input prompt from the frontend
#         input_prompt = request.POST.get("prompt")

#         try:
#             # Call your custom function to get the Gemini response
#             llm_response = gemini_response(input_prompt)

#             # Return response to the frontend
#             return JsonResponse({"response": llm_response})

#         except Exception as e:
#             # Handle any errors from get_gemini_response
#             return JsonResponse({"error": str(e)})

#     return render(request, "genai.html")



from django.shortcuts import render
from django.http import JsonResponse
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input_prompt):
    if input_prompt:
        response=model.generate_content(input_prompt)
        if response:
            return response.text
        else:
            return "No response recived !"
    else:
        return "No Input Prompt recieved"

def gemini_llm_view(request):
    if request.method == 'POST':
        input_prompt=request.POST.get("prompt")

        try :
            llm_response=get_gemini_response(input_prompt)

            return JsonResponse(

                {
                    "response": llm_response
                }
            )
        
        except Exception as e:
            return JsonResponse({"error": str(e)})
        
    return render(request,"genai.html")

