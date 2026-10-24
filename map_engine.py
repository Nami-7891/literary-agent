# map_engine.py
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import os
import requests
import base64
import urllib.parse
import streamlit as st

def generate_huggingface_image(prompt: str, is_macro: bool = False) -> str:
    """Bypasses Hugging Face DNS block using Pollinations.ai (Free SDXL)."""
    
    if is_macro:
        enhanced_prompt = f"Highly detailed top-down vintage map, ink on parchment, clear geography. {prompt}"
    else:
        enhanced_prompt = f"Strict top-down 2D architectural floor plan blueprint, clean white lines on dark blue background, no text. {prompt}"

    # URL encode the prompt so it is safe to pass in a web address
    safe_prompt = urllib.parse.quote(enhanced_prompt)
    
    # Pollinations generates the image on the fly based on the URL
    return f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1000&height=1000&nologo=true"

def initialize_map_state():
    """Sets up the memory caches to prevent redundant API calls."""
    if "current_map_view" not in st.session_state:
        st.session_state.current_map_view = "macro"
    if "cached_macro_image" not in st.session_state:
        st.session_state.cached_macro_image = None
    if "cached_micro_images" not in st.session_state:
        st.session_state.cached_micro_images = {}

def build_interactive_map(map_data: dict):
    """Renders the single-frame, cached map engine."""
    initialize_map_state()
    
    # ---------------------------------------------------------
    # 1. TOP-MOUNTED ESCAPE HATCH (BACK BUTTON)
    # ---------------------------------------------------------
    if st.session_state.current_map_view != "macro":
        if st.button("⬅️ Back to Town Map", use_container_width=True):
            st.session_state.current_map_view = "macro"
            st.rerun()
            
    # ---------------------------------------------------------
    # 2. RENDER MACRO VIEW (THE TOWN MAP)
    # ---------------------------------------------------------
    if st.session_state.current_map_view == "macro":
        # Cache check: Generate only if it doesn't exist
        if not st.session_state.cached_macro_image:
            st.session_state.cached_macro_image = generate_huggingface_image(map_data["town_map_prompt"], is_macro=True)
            
        # Lock the camera bounds to prevent the white/grey void
        m = folium.Map(
            location=[500, 500], 
            zoom_start=1, 
            crs="Simple", 
            tiles=None,
            max_bounds=True,
            min_lat=0, max_lat=1000,
            min_lon=0, max_lon=1000
        )
        
        ImageOverlay(
            image=st.session_state.cached_macro_image,
            bounds=[[0, 0], [1000, 1000]],
            opacity=1,
            interactive=True,
            cross_origin=False,
        ).add_to(m)
        
        # Plot the story-accurate pins
        for landmark in map_data.get("landmarks", []):
            folium.Marker(
                location=landmark["coordinates"],
                popup=landmark["name"],
                tooltip=landmark["marker_tooltip"]
            ).add_to(m)

        map_interaction = st_folium(
            m, 
            width=700, 
            height=500, 
            use_container_width=True, 
            returned_objects=["last_object_clicked"]
        )
        
        # Click Detection Engine: Swap state to micro view if a pin is clicked
        if map_interaction and map_interaction.get("last_object_clicked"):
            clicked = map_interaction["last_object_clicked"]
            
            # Match the clicked Leaflet coordinates (with a small margin for float errors) to the JSON data
            for landmark in map_data.get("landmarks", []):
                lat, lng = landmark["coordinates"]
                if abs(lat - clicked["lat"]) < 1 and abs(lng - clicked["lng"]) < 1:
                    st.session_state.current_map_view = landmark["name"]
                    st.rerun()

    # ---------------------------------------------------------
    # 3. RENDER MICRO VIEW (THE INTERIOR BLUEPRINT)
    # ---------------------------------------------------------
    else:
        active_landmark_name = st.session_state.current_map_view
        
        # Find the specific landmark data
        active_landmark = next((lm for lm in map_data["landmarks"] if lm["name"] == active_landmark_name), None)
        
        if active_landmark:
            st.markdown(f"### 📍 Interior Blueprint: {active_landmark_name}")
            
            # Cache check: Generate interior only if it hasn't been visited yet
            if active_landmark_name not in st.session_state.cached_micro_images:
                st.session_state.cached_micro_images[active_landmark_name] = generate_huggingface_image(active_landmark["blueprint_prompt"], is_macro=False)
                
            # Render the cached blueprint in the exact same frame size
            m_interior = folium.Map(
                location=[500, 500], zoom_start=1, crs="Simple", tiles=None,
                max_bounds=True, min_lat=0, max_lat=1000, min_lon=0, max_lon=1000
            )
            
            ImageOverlay(
                image=st.session_state.cached_micro_images[active_landmark_name],
                bounds=[[0, 0], [1000, 1000]],
                opacity=1,
            ).add_to(m_interior)
            
            st_folium(m_interior, width=700, height=500, use_container_width=True, returned_objects=[])
