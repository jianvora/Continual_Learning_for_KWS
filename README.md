# Continual Learning for Keyword Spotting

This work serves as the course project for CS 753: Automatic Speech Recognition

Significant research has been done in the field of deep learning to add new classes to an existing set of classes in a neural model, and achieve similar performance on the new classes compared to the older classes. We try to extend this idea into the field of speech recognition by considering two separate problem statements. Firstly, we choose the task of joint keyword spotting and speaker identification with the feature of online enrollment at test time, which was proposed in Interspeech'21. Next, we consider the task of continual batch learning for keyword spotting networks, where at each time step we only have access to the current speech data but none of the older data for training. In this setting, we propose a time-efficient solution to ensure that the speech model performance on older training datasets doesn't deteriorate as newer data keeps on coming. 

