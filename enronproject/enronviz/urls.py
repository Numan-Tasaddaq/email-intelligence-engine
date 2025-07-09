from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('story/', views.story, name='story'),
    path('story/<str:story_type>/', views.story_detail, name='story_detail'),
    path('agent/', views.agent, name='agent'),
    path('visualizations/', views.visualization_list, name='visualization_list'),
    path('viz/<str:viz_type>/', views.get_visualization, name='get_visualization'),
     path('stories/all/', views.all_stories, name='all_stories'), 
    path('podcast/', views.generate_podcast, name='generate_podcast'),
    path('chat-with-agent/', views.chat_with_agent, name='chat_with_agent'),


]

