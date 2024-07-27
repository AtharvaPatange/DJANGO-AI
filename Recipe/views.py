# import base64
# from io import BytesIO
# from PIL import Image
# from django.shortcuts import render
# from .forms import RecipeForm
# import google.generativeai as genai
# from django.conf import settings

# # Configure Google Generative AI
# genai.configure(api_key=settings.GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")

# def get_gemini_response(input_prompt, image_str):
#     if input_prompt:
#         response = model.generate_content([input_prompt, image_str])
#     else:
#         response = model.generate_content(image_str)
#     return response.text

# def ats_home(request):
#     if request.method == 'POST':
#         form = RecipeForm(request.POST, request.FILES)
#         if form.is_valid():
#             general_input = form.cleaned_data.get('general_input', '')
#             uploaded_file = form.cleaned_data.get('uploaded_file', None)

#             if uploaded_file:
#                 # Process the uploaded image
#                 image = Image.open(uploaded_file)
#                 buffered = BytesIO()
                
#                 # Correct the format handling
#                 image_format = uploaded_file.name.split('.')[-1].upper()  # Extract the file format
#                 if image_format == 'JPG':
#                     image_format = 'JPEG'  # Correct the format for PIL
                
#                 if image_format not in ['JPEG', 'PNG']:
#                     image_format = 'PNG'  # Fallback to PNG if format is not supported
                
#                 image.save(buffered, format=image_format)
                
#                 image_str = base64.b64encode(buffered.getvalue()).decode()
#             else:
#                 image_str = ""

#             # Check which button was pressed and process accordingly
#             if 'submit_recipe' in request.POST and uploaded_file:
#                 recipe_prompt = '''
#                 You are a Master chef who knows recipes from all over the world: Indian, Italian, American, Russian, Brazilian, etc. Now:
#                 Analyze the uploaded image of the recipe and provide detailed information as follows:
#                 1. **List of Ingredients:** Identify and list all the items visible in the image.
#                 2. **Recipe Instructions:** Generate a step-by-step recipe for preparing the dish shown in the image.
#                 3. **Preparation Method:** Describe the method for making the recipe, including any specific techniques or processes involved.
#                 4. **Precautions:** Mention any precautions or tips that should be considered while preparing the dish.
#                 Ensure the information is clear, comprehensive, and easy to follow.
#                 '''
#                 response = get_gemini_response(recipe_prompt, image_str)
#                 context = {
#                     'form': form,
#                     'recipe_response': response,
#                 }
#                 return render(request, 'app.html', context)

#             elif 'submit_general' in request.POST:
#                 if general_input:
#                     response = get_gemini_response(general_input, "")
#                     context = {
#                         'form': form,
#                         'general_response': response,
#                     }
#                     return render(request, 'app.html', context)
#                 else:
#                     context = {
#                         'form': form,
#                         'error': "Please enter your query or input."
#                     }
#                     return render(request, 'app.html', context)

#     else:
#         form = RecipeForm()

#     return render(request, 'app.html', {'form': form})


import base64
from io import BytesIO
from PIL import Image
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
        response = model.generate_content([input_prompt, image_str])
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

