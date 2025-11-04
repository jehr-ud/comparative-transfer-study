Place Cells → Hipocampo
Grid Cells → Corteza entorrinal medial

=========== Flujo general

      ↓ Entorno
      ↓ Neocórtex (contexto sensorial)
      ↓ Corteza entorrinal (grid cells: coordenadas)
      ↓ Hipocampo (place cells + mapa cognitivo)
      ↓ Corteza prefrontal (planificación, decisión)
      ↓ Neocórtex (consolidación de memoria)
      ↓ Movimiento en el laberinto

Entorno
 ↓
Órganos sensoriales
 ↓
Neocórtex sensorial (visión, tacto, oído)
 ↓
Áreas de asociación multimodal (integración)
 ↓
Corteza entorrinal medial (Grid Cells: coordenadas espaciales)
 ↓
Hipocampo:
  Giro dentado → Pattern separation
  CA3/CA1 → Place Cells (ubicación específica)
 ↓
Corteza prefrontal (planificación y decisión)
 ↓
Corteza motora primaria (acción)



Corteza visual (SNN visual) 
   ↓ tren de spikes
Corteza entorrinal (SNN grid cells)
   ↓ tren de spikes
Hipocampo (SNN place cells + mapa cognitivo)
   ↓ tren de spikes
Corteza prefrontal (SNN de planificación)
   ↓ tren de spikes
Ganglios basales (SNN de refuerzo)
   ↓ tren de spikes
Corteza motora (SNN motora)

========= Exploración (Exploration)
Entorno → Órganos sensoriales → Neocórtex sensorial:

Cuando el organismo está en un entorno nuevo o poco familiar, los órganos sensoriales captan señales diversas (visual, táctil, auditiva).

Estas señales llegan al neocórtex sensorial, donde se procesan y analizan para detectar información relevante del entorno.

Neocórtex sensorial → Áreas de asociación multimodal:

La información sensorial se combina en áreas que integran diferentes modalidades (vista, tacto, sonido) para formar un panorama más completo y contextual.

Áreas de asociación multimodal → Corteza entorrinal medial:

Aquí las Grid Cells codifican coordenadas espaciales formando un sistema de referencia interno basado en la geometría del espacio explorado, creando una especie de mapa en red.

Esta representación espacial permite comenzar a entender la estructura del entorno.

Corteza entorrinal medial → Hipocampo (Giro dentado → CA3/CA1):

En el Giro dentado, ocurre la pattern separation: se distinguen claramente los estímulos similares para no confundir diferentes lugares o experiencias.

En CA3 y CA1, las Place Cells se activan para codificar ubicaciones específicas del entorno, generando un mapa detallado de “dónde estoy”.

Hipocampo → Corteza prefrontal:

La información espacial y contextual llega a la corteza prefrontal, que evalúa posibles rutas, riesgos y recompensas.

En esta etapa se realiza la exploración activa: la corteza prefrontal impulsa la toma de decisiones para probar nuevas rutas o acciones basándose en el aprendizaje y la novedad.

Corteza prefrontal → Corteza motora primaria:

Finalmente, las decisiones tomadas se traducen en órdenes motoras para explorar físicamente el entorno: moverse, tocar, observar nuevas zonas, etc.



======== Explotación (Exploitation)
Cuando ya hay experiencia y conocimiento del entorno:

Percepción del entorno y sensorial → Neocórtex sensorial y áreas de asociación:

Se siguen recibiendo señales, pero ahora el sistema reconoce patrones familiares con rapidez.

Corteza entorrinal medial y Hipocampo:

La información espacial ya está codificada y almacenada.

Las Place Cells y Grid Cells permiten reconocer inmediatamente la ubicación y posibles rutas.

En el giro dentado la necesidad de separar patrones nuevos disminuye porque el entorno es conocido.

Corteza prefrontal:

La toma de decisiones se basa en estrategias conocidas y optimizadas para alcanzar metas, es decir, se explotan las rutas y acciones que previamente han dado buenos resultados.

Se reduce la exploración activa para ahorrar energía y tiempo.

Corteza motora primaria:

Se ejecutan órdenes motoras para moverse con eficacia y seguridad hacia el objetivo, utilizando la información previa almacenada.





EXPLANATION ================ 

16.09.2025

se tiene problema: Temporal Credit Assignment  --> solución Eligibility Traces
el tipo de trasferencia es: domain adaptation o inductive transfer o meta learning o hibrido

                   ┌───────────────────────────┐
                   │      Cortezas Sensoriales │
                   │ (Visual, Auditiva, Somato)│
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │  Cortezas Asociativas      │
                   │ (Integración Multimodal)  │
                   └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼─────────────────────────────┐
          ▼                      ▼                             ▼
┌───────────────────┐  ┌───────────────────┐         ┌──────────────────────┐
│   Hipocampo        │  │ Corteza Prefrontal│         │   Neuromodulación    │
│ Memoria episódica  │  │ Control ejecutivo │         │ (dopamina, ACh, etc.)│
└──────┬─────────────┘  └───────────┬──────┘         └──────────┬───────────┘
       │                            │                           │
       │                            │                           │
       ▼                            ▼                           ▼
┌───────────────────┐     ┌───────────────────┐        ┌───────────────────┐
│ Replaying secuencias│   │ Planificación y   │        │ Ajuste de plasticidad│
│ episódicas          │   │ toma de decisiones│        │ y priorización      │
└──────┬─────────────┘   └───────────┬───────┘        └───────────────────┘
       │                             │
       ▼                             ▼
┌──────────────────────────────────────────────────────┐
│                    Neocórtex                         │
│ Aprendizaje lento, consolidación semántica,           │
│ generalización de patrones (a partir del hipocampo).  │
└───────────┬───────────────────────────┬─────────────┘
            │                           │
            ▼                           ▼
  ┌───────────────────┐        ┌────────────────────────┐
  │ Conocimiento      │        │   Cortezas Motoras     │
  │ abstracto / reglas│        │ Generación de acción   │
  └───────────────────┘        │ en el entorno Gym      │
                               └────────────────────────┘


[ Entrada sensorial ]
        │
        ▼
[ Cortezas sensoriales primarias ]
        │  (visión, audición, tacto, etc.)
        ▼
[ Cortezas asociativas ]
        │  (integración multimodal, significado)
        ▼
 ┌─────────────────────────────────────────┐
 │        Consulta con la memoria           │
 │ ┌──────────────┐   ┌─────────────────┐ │
 │ │  Hipocampo   │ ↔ │   Neocortex     │ │
 │ │ (episódica)  │   │ (semántica)     │ │
 │ └──────────────┘   └─────────────────┘ │
 │          ↕ (puente)                     │
 │    Corteza entorrinal                   │
 └─────────────────────────────────────────┘
        │
        ▼
[ Corteza prefrontal ]
        │  (planificación, predicción, razonamiento)
        ▼
[ Ganglios basales ]
        │  (selección de acción, filtro de valor/recompensa)
        ▼
[ Corteza motora ]
        │  (ejecución de la acción)
        ▼
[ Acción realizada ]
        │
        ▼
[ Feedback / Aprendizaje ]
        │  (refuerzo positivo/negativo, actualización de memoria)
        └───────────────────────────────┐
                                        │
       <──────── Retroalimentación ─────┘


proximo paso: definir una arquitectura basica de cortezas que genere memoria semantica y memoria episodica
            definir si se evalua cada corteza


17.9.25 
lo ideal es que el agente pueda reconocer subtareas para poder trasfereirse en la memoria. 
Se puede usar el algoritmo DIAYN

La idea central: un agente puede aprender comportamientos útiles aunque no le des ninguna recompensa explícita del entorno.