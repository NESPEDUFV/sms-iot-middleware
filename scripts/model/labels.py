from collections import namedtuple

Label = namedtuple("Label", ("long", "short"))

TOPIC_LABELS = {
    "iiot": Label("Industrial IoT", "IIoT"),
    "wsn": Label("Wireless Sensor Networks", "WSNs"),
    "agriculture": Label("Smart Agriculture", "Smart Agriculture"),
    "multimedia": Label("Multimedia", "Multimedia"),
    "general": Label("General", "General"),
    "sdn": Label("Software Defined Networks", "SDNs"),
    "energy": Label("Energy", "Energy"),
    "context": Label("Context Awareness", "Context"),
    "aal": Label("Ambient Assisted Living", "AAL"),
    "discovery": Label("Resource Discovery", "Res. Discovery"),
    "city": Label("Smart City", "Smart City"),
}

topic_labels = {
    "iiot": "Industrial IoT",
    "wsn": "Wireless Sensor Networks",
    "agriculture": "Smart Agriculture",
    "multimedia": "Multimedia",
    "general": "General",
    "sdn": "Software Defined Networks",
    "energy": "Energy",
    "context": "Context Awareness",
    "aal": "Ambient Assisted Living",
    "discovery": "Resource Discovery",
    "city": "Smart City",
}

topic_short_labels = {
    "iiot": "IIoT",
    "wsn": "WSNs",
    "agriculture": "Smart Agriculture",
    "multimedia": "Multimedia",
    "general": "General",
    "sdn": "SDNs",
    "energy": "Energy",
    "context": "Context",
    "aal": "AAL",
    "discovery": "Res. Discovery",
    "city": "Smart City",
}

METHOD_LABELS = {
    "adhoc": Label("Ad Hoc", "Ad Hoc"),
    "slr": Label("Systematic Literature Review", "SLR"),
    "sms": Label("Systematic Mapping Study", "SMS"),
}

method_labels = {
    "adhoc": "Ad Hoc",
    "slr": "Systematic Literature Review",
    "sms": "Systematic Mapping Study",
}

method_short_labels = {
    "adhoc": "Ad Hoc",
    "slr": "SLR",
    "sms": "SMS",
}

ASPECT_LABELS = {
    "architecture": Label("Architecture", "Architecture"),
    "approach": Label("Design", "Design"),
    "requirements": Label("Requirements", "Requirements"),
    "interoperability": Label("Interoperability", "Interoperability"),
    "security": Label("Security & Privacy", "Security"),
    "efficiency": Label("Efficiency", "Efficiency"),
    "protocols": Label("Communication Protocols", "Protocols"),
    "implementation": Label("Implementation", "Implementation"),
    "evaluation": Label("Evaluation", "Evaluation"),
    "context": Label("Context Awareness", "Context"),
    "ai": Label("Artificial Intelligence", "AI"),
}

aspect_labels = {
    "architecture": "Architecture",
    "approach": "Design",
    "requirements": "Requirements",
    "interoperability": "Interoperability",
    "security": "Security & Privacy",
    "efficiency": "Efficiency",
    "protocols": "Communication Protocols",
    "implementation": "Implementation",
    "evaluation": "Evaluation",
    "context": "Context Awareness",
    "ai": "Artificial Intelligence",
}

aspect_short_labels = {
    "architecture": "Architecture",
    "approach": "Design",
    "requirements": "Requirements",
    "interoperability": "Interoperability",
    "security": "Security",
    "efficiency": "Efficiency",
    "protocols": "Protocols",
    "implementation": "Implementation",
    "evaluation": "Evaluation",
    "context": "Context",
    "ai": "AI",
}
