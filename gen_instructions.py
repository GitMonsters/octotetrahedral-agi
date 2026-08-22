import json
import random
import itertools

random.seed(42)

CATEGORIES = {
    "science": {
        "topics": ["physics", "chemistry", "biology", "astronomy", "ecology", "genetics", "quantum mechanics", "thermodynamics", "evolution", "neuroscience", "geology", "meteorology", "optics", "acoustics", "electromagnetism", "organic chemistry", "molecular biology", "astrophysics", "paleontology", "microbiology"],
        "aspects": ["fundamental principles", "key discoveries", "real world applications", "historical development", "modern research", "core theories", "practical implications", "major breakthroughs", "underlying mechanisms", "future directions"],
    },
    "technology": {
        "topics": ["artificial intelligence", "machine learning", "blockchain", "cloud computing", "cybersecurity", "internet of things", "quantum computing", "robotics", "virtual reality", "augmented reality", "5G networks", "edge computing", "natural language processing", "computer vision", "data science", "devops", "microservices", "containerization", "serverless computing", "autonomous vehicles"],
        "aspects": ["core concepts", "practical applications", "key benefits", "current trends", "underlying technology", "industry impact", "challenges and limitations", "future potential", "best practices", "real world examples"],
    },
    "history": {
        "topics": ["ancient Rome", "World War II", "the Renaissance", "the Industrial Revolution", "ancient Egypt", "the French Revolution", "the Cold War", "the Silk Road", "the Age of Exploration", "the Scientific Revolution", "ancient Greece", "the Middle Ages", "the American Revolution", "the Enlightenment", "the Agricultural Revolution", "the Byzantine Empire", "the Mongol Empire", "the Ottoman Empire", "colonialism", "decolonization"],
        "aspects": ["major events", "key figures", "lasting impact", "causes and effects", "cultural significance", "political changes", "economic transformations", "technological advances", "social movements", "legacy and influence"],
    },
    "mathematics": {
        "topics": ["algebra", "geometry", "calculus", "statistics", "number theory", "linear algebra", "probability", "topology", "differential equations", "combinatorics", "graph theory", "logic", "set theory", "fractals", "chaos theory", "game theory", "cryptography", "numerical analysis", "category theory", "information theory"],
        "aspects": ["key concepts", "real world applications", "historical development", "fundamental theorems", "problem solving approaches", "connections to other fields", "computational methods", "practical uses", "beautiful results", "open problems"],
    },
    "philosophy": {
        "topics": ["ethics", "logic", "epistemology", "metaphysics", "aesthetics", "political philosophy", "existentialism", "stoicism", "utilitarianism", "nihilism", "phenomenology", "pragmatism", "rationalism", "empiricism", "the philosophy of mind", "philosophy of language", "philosophy of science", "Buddhist philosophy", "Confucian philosophy", "African philosophy"],
        "aspects": ["core ideas", "key thinkers", "modern applications", "major debates", "practical implications", "historical context", "contemporary relevance", "thought experiments", "critiques and counterarguments", "influence on society"],
    },
    "health": {
        "topics": ["nutrition", "exercise", "sleep", "mental health", "immune system", "cardiovascular health", "bone health", "hygiene", "meditation", "yoga", "first aid", "vaccination", "allergies", "chronic disease prevention", "healthy aging", "gut health", "brain health", "eye health", "dental care", "respiratory health"],
        "aspects": ["best practices", "common myths", "scientific evidence", "practical tips", "warning signs", "prevention strategies", "treatment options", "lifestyle changes", "research findings", "daily habits"],
    },
    "environment": {
        "topics": ["climate change", "renewable energy", "biodiversity", "pollution", "deforestation", "ocean conservation", "sustainable agriculture", "recycling", "carbon footprint", "water conservation", "wildlife protection", "air quality", "soil conservation", "green technology", "environmental policy", "urban sustainability", "waste management", "coral reef protection", "freshwater ecosystems", "air pollution"],
        "aspects": ["current challenges", "solutions", "scientific evidence", "individual actions", "policy approaches", "technological innovations", "ecological impact", "economic implications", "global efforts", "future outlook"],
    },
    "language": {
        "topics": ["grammar", "writing", "public speaking", "foreign languages", "linguistics", "poetry", "storytelling", "technical writing", "persuasive writing", "creative writing", "etymology", "syntax", "semantics", "phonetics", "sociolinguistics", "translation", "debate", "communication skills", "media literacy", "critical thinking"],
        "aspects": ["fundamentals", "advanced techniques", "common mistakes", "practice exercises", "professional applications", "creative approaches", "historical evolution", "cross cultural aspects", "digital age skills", "learning strategies"],
    },
    "daily_life": {
        "topics": ["cooking", "financial planning", "time management", "home organization", "travel planning", "gardening", "parenting", "relationship building", "stress management", "productivity", "basic repair skills", "meal preparation", "personal finance", "goal setting", "self improvement", "negotiation", "decision making", "networking", "work life balance", "minimalism"],
        "aspects": ["practical tips", "common mistakes", "expert advice", "beginner guide", "advanced strategies", "daily routines", "essential skills", "time saving techniques", "budget friendly approaches", "long term benefits"],
    },
    "arts": {
        "topics": ["music", "visual arts", "literature", "film", "architecture", "dance", "theater", "photography", "sculpture", "graphic design", "interior design", "fashion", "animation", "game design", "digital art", "pottery", "calligraphy", "comics", "street art", "opera"],
        "aspects": ["history", "techniques", "famous works", "modern trends", "cultural significance", "creative process", "appreciation guide", "getting started", "notable artists", "impact on society"],
    },
}

FACTS = {
    "science": [
        "The human body contains about 37.2 trillion cells, each performing specialized functions.",
        "Black holes can have millions to billions of times the mass of our Sun.",
        "CRISPR gene editing technology allows precise modifications to DNA in living organisms.",
        "The ozone layer absorbs most of the Sun's ultraviolet radiation, protecting life on Earth.",
        "Mitochondria are often called the powerhouses of the cell because they generate most of the cell's energy.",
        "The average human brain contains approximately 86 billion neurons.",
        "Sound travels at about 343 meters per second in air at room temperature.",
        "Water is the only common substance that is less dense as a solid than as a liquid.",
        "The universe is estimated to be about 13.8 billion years old.",
        "Plate tectonics theory explains how Earth's outer shell is divided into several plates that move over the mantle.",
        "Electromagnetic waves include radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays.",
        "Photosynthesis converts about 1% of sunlight energy into chemical energy in plants.",
        "The speed of gravity is the same as the speed of light according to general relativity.",
        "Antibiotics are ineffective against viral infections like the common cold or flu.",
        "DNA replication occurs with remarkable accuracy, with only about one error per billion nucleotides copied.",
    ],
    "technology": [
        "The first computer programmer was Ada Lovelace, who wrote algorithms for Charles Babbage's analytical engine.",
        "Machine learning models improve by adjusting internal parameters to minimize prediction errors.",
        "The Internet was originally developed as ARPANET, a US Department of Defense project in the 1960s.",
        "Python is one of the most popular programming languages due to its readability and versatility.",
        "Open source software allows anyone to view, modify, and distribute the source code.",
        "APIs enable different software applications to communicate and share data with each other.",
        "Deep learning uses neural networks with many layers to learn hierarchical data representations.",
        "Git is a distributed version control system that tracks changes in source code during development.",
        "Docker containers package applications with their dependencies for consistent deployment across environments.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors.",
        "WebAssembly allows code written in multiple languages to run in web browsers at near-native speed.",
        "Neural network architectures like transformers have revolutionized natural language processing.",
        "Edge computing processes data closer to where it is generated, reducing latency and bandwidth usage.",
        "Blockchain creates immutable records through cryptographic hashing and distributed consensus.",
    ],
    "history": [
        "The Library of Alexandria was one of the largest and most significant libraries of the ancient world.",
        "The printing press invented by Gutenberg around 1440 revolutionized the spread of knowledge.",
        "The Renaissance began in Italy and spread throughout Europe, transforming art, science, and philosophy.",
        "The American Constitution, ratified in 1788, established the framework for the US government.",
        "The Agricultural Revolution around 10,000 BC enabled humans to transition from nomadic to settled life.",
        "The Black Death killed an estimated 75 to 200 million people in Eurasia during the 14th century.",
        "The Rosetta Stone, discovered in 1799, was key to deciphering Egyptian hieroglyphs.",
        "The Wright brothers achieved the first sustained powered flight in 1903.",
        "The United Nations was founded in 1945 to promote international cooperation and prevent conflicts.",
        "The Berlin Wall fell in 1989, symbolizing the end of the Cold War division of Europe.",
        "Ancient Mesopotamia is often called the cradle of civilization for its early urban development.",
        "The Magna Carta signed in 1215 established the principle that everyone is subject to the law.",
        "The Space Race between the US and USSR accelerated technological development in the 20th century.",
        "The abolition of slavery was achieved through movements spanning centuries across many countries.",
        "The Digital Revolution beginning in the late 20th century transformed how humans communicate and work.",
    ],
    "mathematics": [
        "The number zero was independently invented in several ancient civilizations.",
        "Euler's identity e raised to i pi plus one equals zero connects five fundamental mathematical constants.",
        "The Fibonacci sequence appears in nature, from flower petals to spiral galaxies.",
        "Statistics is the science of learning from data and making informed decisions under uncertainty.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        "Prime numbers are the building blocks of all natural numbers through multiplication.",
        "Probability theory provides a mathematical framework for quantifying uncertainty.",
        "Linear algebra is essential for computer graphics, machine learning, and quantum mechanics.",
        "Calculus was independently developed by Newton and Leibniz in the 17th century.",
        "The golden ratio approximately 1.618 appears in art, architecture, and nature.",
        "Topology studies properties of shapes that remain unchanged under continuous deformation.",
        "Graph theory originated from Euler's solution to the seven bridges of Konigsberg problem.",
        "Fractals are infinitely complex patterns that are self similar across different scales.",
        "Game theory analyzes strategic interactions between rational decision makers.",
        "The Mandelbrot set is one of the most famous objects in mathematics, displaying infinite complexity.",
    ],
    "default": [
        "Knowledge builds upon itself, with each discovery opening doors to new questions and understanding.",
        "Critical thinking involves analyzing information objectively and making reasoned judgments.",
        "Effective communication is one of the most valuable skills in both personal and professional life.",
        "Continuous learning helps individuals adapt to changing circumstances and expand their capabilities.",
        "Creativity involves connecting ideas in novel ways to produce original and valuable outcomes.",
        "Collaboration combines diverse perspectives and skills to achieve goals that individuals cannot reach alone.",
        "Empathy, the ability to understand and share others feelings, is fundamental to human relationships.",
        "Sustainability means meeting present needs without compromising future generations ability to meet theirs.",
        "Resilience is the capacity to recover quickly from difficulties and adapt to challenging situations.",
        "Innovation drives progress by finding better solutions to existing problems and creating new possibilities.",
    ],
}

INSTRUCTION_TEMPLATES = [
    "What is {topic}?",
    "Explain {topic}.",
    "How does {topic} work?",
    "Tell me about {topic}.",
    "Why is {topic} important?",
    "Describe {topic}.",
    "What are the key concepts of {topic}?",
    "Give me an overview of {topic}.",
    "What should I know about {topic}?",
    "How can I learn about {topic}?",
    "What are the main ideas behind {topic}?",
    "Why do people study {topic}?",
    "What makes {topic} interesting?",
    "How is {topic} used in practice?",
    "What are the applications of {topic}?",
]

CHAT_PAIRS = [
    ("Hello!", "Hello! I am happy to help you with any questions you have. What would you like to know about?"),
    ("Hi there!", "Hi! Ask me anything about science, technology, history, math, or many other topics."),
    ("How are you?", "I am doing well, thank you for asking! How can I assist you today?"),
    ("Who are you?", "I am an AI assistant built with the OctoTetrahedral architecture. I can answer questions on many topics."),
    ("What can you do?", "I can answer questions about science, technology, history, math, philosophy, arts, and many other subjects."),
    ("Tell me a joke.", "Why did the AI go to therapy? Because it had too many deep learning issues!"),
    ("Thank you.", "You are welcome! Feel free to ask if you have more questions."),
    ("Thanks!", "You are welcome! Let me know if there is anything else I can help with."),
    ("Goodbye!", "Goodbye! Have a great day, and come back anytime you need help."),
    ("What is the meaning of life?", "The meaning of life is a profound philosophical question. Different perspectives offer different answers, from pursuing happiness to contributing to society to seeking knowledge."),
    ("Do you have feelings?", "I am an AI language model and do not have feelings or consciousness. I am designed to process and generate text to be helpful."),
    ("What is your favorite color?", "As an AI, I do not have personal preferences, but I find color fascinating from both physics and psychology perspectives."),
    ("Can you help me?", "Of course! I would be happy to help. What topic or question do you have in mind?"),
    ("Tell me something interesting.", "Did you know that honey never spoils? Archaeologists have found 3000 year old honey in Egyptian tombs that was still perfectly edible."),
    ("What is love?", "Love is a complex emotion involving deep affection, attachment, and care for another person. It manifests in many forms including romantic, familial, and platonic love."),
    ("How do I become smarter?", "Reading widely, staying curious, exercising regularly, getting adequate sleep, and engaging in deep thinking are all proven ways to enhance cognitive abilities."),
    ("What is the most important invention?", "This is debated, but the printing press, electricity, and the internet are often cited as among the most transformative inventions in human history."),
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth atmosphere scatter shorter wavelength blue light more than other colors, a phenomenon known as Rayleigh scattering."),
    ("How far away is the Moon?", "The Moon is approximately 384,400 kilometers from Earth on average. It takes about 3 days for spacecraft to travel there."),
    ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in Earth crust due to tectonic plate movement, creating seismic waves that shake the ground."),
]


def generate():
    lines = []
    seen = set()

    for category, data in CATEGORIES.items():
        facts = FACTS.get(category, FACTS["default"])
        for topic in data["topics"]:
            num_templates = random.randint(3, 6)
            templates = random.sample(INSTRUCTION_TEMPLATES, num_templates)
            for template in templates:
                instruction = template.format(topic=topic)
                fact = random.choice(facts)
                aspect = random.choice(data["aspects"])
                response = f"{fact} In terms of {aspect}, this field continues to evolve with new discoveries and applications."
                key = (instruction, topic)
                if key not in seen:
                    seen.add(key)
                    lines.append({"text": f"Instruction: {instruction}\nResponse: {response}"})

    for q, a in CHAT_PAIRS:
        lines.append({"text": f"Instruction: {q}\nResponse: {a}"})
        lines.append({"text": f"Instruction: {q.lower()}\nResponse: {a}"})

    question_words = ["what", "how", "why", "when", "where", "who"]
    connectors = ["and", "but", "or", "also", "additionally", "moreover"]
    for _ in range(500):
        cat1, cat2 = random.sample(list(CATEGORIES.keys()), 2)
        t1 = random.choice(CATEGORIES[cat1]["topics"])
        t2 = random.choice(CATEGORIES[cat2]["topics"])
        qtype = random.choice(question_words)
        if qtype == "what":
            inst = f"What is the relationship between {t1} and {t2}?"
            resp = f"The relationship between {t1} and {t2} involves interesting connections. {t1} provides foundational concepts that can be applied to understanding {t2}."
        elif qtype == "how":
            inst = f"How is {t1} related to {t2}?"
            resp = f"{t1} and {t2} are related through shared principles and methodologies. Advances in one often influence progress in the other."
        else:
            inst = f"Why is {t1} important for {t2}?"
            resp = f"{t1} is important for {t2} because it provides essential foundations. Understanding {t1} enhances our ability to work effectively with {t2}."
        lines.append({"text": f"Instruction: {inst}\nResponse: {resp}"})

    for _ in range(200):
        cat = random.choice(list(CATEGORIES.keys()))
        t = random.choice(CATEGORIES[cat]["topics"])
        inst = f"Give me a detailed explanation of {t}."
        f1 = random.choice(FACTS.get(cat, FACTS["default"]))
        f2 = random.choice(FACTS.get(cat, FACTS["default"]))
        while f2 == f1:
            f2 = random.choice(FACTS.get(cat, FACTS["default"]))
        resp = f"{f1} Additionally, {f2} Understanding these aspects provides a solid foundation for further learning."
        lines.append({"text": f"Instruction: {inst}\nResponse: {resp}"})

    for _ in range(200):
        cat = random.choice(list(CATEGORIES.keys()))
        topics = random.sample(CATEGORIES[cat]["topics"], 2)
        inst = f"Compare {topics[0]} and {topics[1]}."
        resp = f"Both {topics[0]} and {topics[1]} are important areas within {cat}. While {topics[0]} focuses on specific principles, {topics[1]} approaches related challenges from a different angle. Studying both provides a well rounded understanding."
        lines.append({"text": f"Instruction: {inst}\nResponse: {resp}"})

    random.shuffle(lines)
    return lines


if __name__ == "__main__":
    import os
    lines = generate()
    os.makedirs("data", exist_ok=True)
    with open("data/instructions.jsonl", "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(lines)} instruction/response pairs")
