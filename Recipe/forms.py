# ai_recipe_app/forms.py
from django import forms
import streamlit as st



class RecipeForm(forms.Form):
    general_input = forms.CharField(label="Ask or Input", max_length=255, required=False)
    uploaded_file = forms.ImageField(label="Upload an image", required=False)

