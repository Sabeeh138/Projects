from django.urls import path
from . import views

urlpatterns = [
    path('', views.default, name='default'),
    path('chat/', views.chat, name='chat'),
    path('form/', views.form, name='form'),
    path('old/', views.index, name='index'),  # Keep old page for reference
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/form-recommendation/', views.form_recommendation_api, name='form_recommendation_api'),
]
