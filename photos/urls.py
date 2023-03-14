from django.urls import path
from . import views

urlpatterns = [
    path('', views.blurPhoto, name='blur'),
]
