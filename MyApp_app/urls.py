from django.urls import path
from . import views

urlpatterns = [
    path('', views.text_input, name='textinput'),
    path('login/', views.Login.as_view(), name='login'),   
    path('logout/', views.Logout.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    ]