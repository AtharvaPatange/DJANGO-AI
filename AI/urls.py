"""
URL configuration for AI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# ai_recipe_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('Recipe.urls')),
]
from django.urls import path
from Recipe.views import gemini,image,diet,index,process_request,gemini_llm_view
from django.views.generic import TemplateView

urlpatterns = [
    # path('', index, name='index'),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('text/', gemini, name='ats_home'),
    path('image/', image ,name='chat'),
    path('diet/', diet ,name='chat'),
    path('index/', index , name='index'),
    
    path('AI/',process_request,name='AI'),
    path('llm/', gemini_llm_view , name="llm")
   
]



urlpatterns+= staticfiles_urlpatterns()