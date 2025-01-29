This was originally supposed to be  implemented with groq/together.ai's AI inference, similar to the LLM in my [portfolio](avdportfolio.vercel.app/chat) that i deployed like 1 week and a half ago. I tested it with some sample data and the results were good but there are some noteworthy limitations of using inference here 
- First of all, such a long system message is UNREALISTIC, as it would have to contain info about curriculum, rules, and more.
- Also, that much info, if somehow fit into a singular system message, would mean smaller models would be more prone to inacurate responses (context window). And small models are usually the ones given for free.

Hence, I decided to go with RAG here. I had already known about and was looking into RAG models for a while before I decided to make this. So after some learning and tutorials, I succesfully implemented an extremely simple application.
<br><br>
Have a great day! Aravind
