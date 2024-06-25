from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_currency, name='detect_currency'),
]
