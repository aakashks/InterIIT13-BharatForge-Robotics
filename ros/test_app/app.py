import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json
import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pydantic import BaseModel
from typing import List

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from llm import get_possible_objects
from vision import run_clip_on_objects, run_vlm

from icecream import ic


# Global variable to track ROS2 initialization
if "ros_initialized" not in st.session_state:
    st.session_state.ros_initialized = False

class ROS2Interface(Node):
    def __init__(self):
        super().__init__('streamlit_interface')
        self.publisher_ = self.create_publisher(String, 'coordinate_data', 10)
        self.subscription = self.create_subscription(
            String,
            'ros_feedback',
            self.listener_callback,
            10
        )
        self.received_message = ''

    def listener_callback(self, msg):
        self.received_message = msg.data

    def publish(self, coord_data):        #
        msg = String()
        msg.data = json.dumps(coord_data)
        ic(msg)
        self.publisher_.publish(msg)


# Initialize ROS2 only once
if not st.session_state.ros_initialized:
    rclpy.init()
    st.session_state.ros_node = ROS2Interface()
    st.session_state.ros_initialized = True

ros_node = st.session_state.ros_node


db_client = chromadb.PersistentClient('/home/user1/s_ws/.chromadb_cache')
embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli', device='cuda')
collection = db_client.get_collection('test1', embedding_function=embedding_function, data_loader=ImageLoader())


st.title("🔍 Object Identifier with ROS2 Integration")

st.write(
    """
    Enter a command or query, and the system will identify possible objects or entities 
    that a robot should interact with based on the input. The identified objects will be 
    published to ROS2 topic 'identified_objects'.
    """
)

# Text input for the user's prompt
prompt = st.text_input("Enter your command or query:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Submit") and prompt:
        with st.spinner("Processing..."):
            try:
                # Get possible objects from the user query
                objects_json = get_possible_objects(prompt)
                st.success("Possible Objects Identified:")
                st.json(objects_json)
                
                # extract list of objects
                object_list = objects_json['possible_objects']
                ic(object_list)
                obj_path = run_clip_on_objects(object_list, collection)
                ic(obj_path)
                coord_data = run_vlm(obj_path)
                
                ros_node.publish(coord_data)
                
            except ValueError as ve:
                st.error(f"Failed to process the query: {ve}")

with col2:
    # Display received messages from robot
    if st.button("Check Robot Feedback"):
        rclpy.spin_once(ros_node, timeout_sec=0.1)
        if ros_node.received_message:
            st.write("Robot Feedback:", ros_node.received_message)
        else:
            st.write("No feedback received from robot")

# Add a section to display ROS2 status
st.sidebar.title("ROS2 Status")
st.sidebar.write("Connection Status: Active" if st.session_state.ros_initialized else "Connection Status: Inactive")
if st.sidebar.button("Reset ROS2 Connection"):
    st.session_state.ros_initialized = False
    st.experimental_rerun()

