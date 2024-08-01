# ai_recipe_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ats_home, name='ats_home'),
    path('/chat',views.chat_with_gemini, name='chat')
]
