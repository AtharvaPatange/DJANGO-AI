# ai_recipe_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ats_home, name='ats_home'),
]
