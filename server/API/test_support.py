import requests
import json

test_cases = {
    "cases": [
        {
            "question": "Hola",
            "answer": "Hola, ¿en qué puedo ayudarte? Por favor, proporciona más detalles sobre tu consulta como tu número de orden o correo electrónico.",
            "category": "saludo",
            "priority": 1
        },
        {
            "question": "Mi número de orden es ORD123456",
            "answer": "He encontrado su orden ORD123456. Estado actual: En tránsito\nFecha estimada de entrega: 25 de febrero\nÚltima actualización: Paquete en centro de distribución local\n¿Necesita información adicional sobre su pedido?",
            "category": "seguimiento_pedido",
            "priority": 1
        },
        {
            "question": "Mi correo es usuario@ejemplo.com",
            "answer": "He encontrado las siguientes órdenes asociadas a usuario@ejemplo.com:\n1. ORD123456 - En tránsito\n2. ORD123457 - Entregado\n3. ORD123458 - Procesando\n¿Sobre cuál orden necesita más información?",
            "category": "seguimiento_pedido",
            "priority": 1
        },
        {
            "question": "Quiero saber sobre la orden ORD123456",
            "answer": "Detalles de la orden ORD123456:\n- Fecha de compra: 20/02/2024\n- Estado: En tránsito\n- Productos: 2 artículos\n- Último evento: Llegada a centro de distribución\n- Próxima actualización estimada: 24 horas\n¿Necesita información específica sobre algún aspecto de su orden?",
            "category": "seguimiento_detallado",
            "priority": 1
        },
        {
            "question": "¿Cómo puedo rastrear mi pedido?",
            "answer": "Para rastrear su pedido, necesito su número de orden y correo electrónico. Con gusto le ayudaré a localizar su paquete.",
            "category": "seguimiento_pedido",
            "priority": 1
        },
        {
            "question": "Me cobraron dos veces por mi pedido",
            "answer": "Lamento el inconveniente. ¿Podría proporcionarme su número de orden y los últimos cuatro dígitos del método de pago utilizado? Revisaré los detalles de la transacción.",
            "category": "problemas_facturacion",
            "priority": 1
        },
        {
            "question": "Mi pedido está retrasado, ¿cuándo lo recibiré?",
            "answer": "Entiendo su preocupación. Permítame verificar los últimos detalles de seguimiento de su pedido. ¿Puede compartir su número de orden?",
            "category": "seguimiento_pedido",
            "priority": 1
        },
        {
            "question": "Recibí el artículo equivocado",
            "answer": "¡Lamento la confusión! ¿Podría proporcionarme su número de orden y una foto del artículo incorrecto? Organizaremos un reemplazo o reembolso.",
            "category": "problemas_pedido",
            "priority": 2
        },
        {
            "question": "¿Cómo puedo descargar mi factura?",
            "answer": "Puede descargar su factura desde su cuenta en la sección 'Historial de Pedidos'. Si necesita ayuda, proporcione su número de orden y puedo enviársela.",
            "category": "solicitud_factura",
            "priority": 3
        },
        {
            "question": "Necesito actualizar mi método de pago",
            "answer": "Para actualizar su método de pago, vaya a la configuración de su cuenta en 'Facturación'. Si necesita ayuda, puedo guiarlo en el proceso.",
            "category": "problemas_facturacion",
            "priority": 2
        },
        {
            "question": "¿Cuánto tarda un reembolso?",
            "answer": "Los reembolsos típicamente tardan de 5 a 7 días hábiles en procesarse, dependiendo de su banco. ¿Desea que verifique el estado de su reembolso?",
            "category": "reembolsos",
            "priority": 3
        },
        {
            "question": "¿Puedo cambiar la dirección de envío después de hacer el pedido?",
            "answer": "Si su pedido aún no se ha enviado, es posible que pueda actualizar la dirección. ¿Puede proporcionarme su número de orden para verificar?",
            "category": "cambios_pedido",
            "priority": 2
        },
        {
            "question": "¿Ofrecen planes de pago o pagos en cuotas?",
            "answer": "Sí, ofrecemos planes de pago a través de [proveedor de pagos]. ¿Le gustaría ver las opciones disponibles para su pedido?",
            "category": "opciones_pago",
            "priority": 3
        },
        {
            "question": "Necesito un reembolso",
            "answer": "Entiendo que solicita un reembolso. ¿Podría proporcionarme su número de orden y el motivo de la solicitud?",
            "category": "reembolsos",
            "priority": 2
        },
        {
            "question": "¿Pueden enviarme una copia de mi factura?",
            "answer": "Con gusto. ¿Podría proporcionarme su número de orden o su correo electrónico para localizar su factura y enviársela?",
            "category": "solicitud_factura",
            "priority": 2
        },
        {
            "question": "Hice un pedido pero no recibí la confirmación por correo",
            "answer": "Entiendo la preocupación. ¿Podría proporcionarme su número de orden o correo electrónico para verificar el estado de su compra?",
            "category": "confirmacion_pedido",
            "priority": 1
        },
        {
            "question": "El estado de mi pago dice 'Pendiente', ¿qué significa?",
            "answer": "Un estado 'Pendiente' significa que el pago aún no ha sido confirmado por su banco. Puede tardar hasta 24 horas en procesarse. ¿Desea que revise el estado en nuestro sistema?",
            "category": "problemas_facturacion",
            "priority": 2
        },
        {
            "question": "Mi pago fue rechazado, pero el dinero fue descontado",
            "answer": "Lamento la situación. A veces, los bancos retienen temporalmente los fondos. ¿Podría proporcionarme los últimos cuatro dígitos de su tarjeta o el ID de la transacción para revisar el estado del pago?",
            "category": "problemas_facturacion",
            "priority": 1
        },
        {
            "question": "Necesito modificar los datos de mi factura",
            "answer": "Puedo ayudarle con eso. ¿Podría indicarme su número de orden y qué información necesita corregir?",
            "category": "solicitud_factura",
            "priority": 2
        },
        {
            "question": "Hice un pedido, pero quiero cancelarlo",
            "answer": "Podemos verificar si su pedido aún puede ser cancelado. ¿Podría proporcionarme su número de orden?",
            "category": "cancelacion_pedido",
            "priority": 1
        },
        {
            "question": "Me cobraron un monto incorrecto",
            "answer": "Lamento el inconveniente. ¿Podría indicarme su número de orden y el monto cobrado para verificarlo?",
            "category": "problemas_facturacion",
            "priority": 1
        },
        {
            "question": "Mi tarjeta fue rechazada, pero tiene saldo disponible",
            "answer": "Puede ser un problema con la autorización del banco. ¿Podría probar otro método de pago o contactarse con su banco? Si necesita ayuda, estoy aquí para guiarle.",
            "category": "problemas_pago",
            "priority": 2
        },
        {
            "question": "¿Puedo pagar con PayPal?",
            "answer": "Sí, aceptamos pagos a través de PayPal. Puede seleccionarlo como su método de pago en el proceso de compra.",
            "category": "opciones_pago",
            "priority": 3
        },
        {
            "question": "¿Puedo obtener un comprobante de pago?",
            "answer": "Sí, podemos enviarle el comprobante. ¿Podría proporcionarme su número de orden o el ID de la transacción?",
            "category": "solicitud_factura",
            "priority": 2
        },
        {
            "question": "El código de seguimiento de mi pedido no funciona",
            "answer": "Lamento el problema. A veces, los códigos tardan unas horas en activarse. ¿Podría indicarme su número de orden para verificar el estado?",
            "category": "seguimiento_pedido",
            "priority": 2
        },
        {
            "question": "Mi pedido aparece como entregado, pero no lo recibí",
            "answer": "¡Lamento escuchar eso! ¿Podría indicarme su número de orden? Verificaremos la ubicación exacta de la entrega.",
            "category": "problemas_pedido",
            "priority": 1
        },
        {
            "question": "¿Ofrecen facturación electrónica?",
            "answer": "Sí, ofrecemos facturación electrónica. Puede descargarla desde su cuenta o solicitarla aquí con su número de orden.",
            "category": "solicitud_factura",
            "priority": 3
        },
        {
            "question": "¿Cómo solicito un reembolso si pagué con PayPal?",
            "answer": "Si su pago fue realizado con PayPal, su reembolso se procesará a la misma cuenta en un plazo de 5-7 días hábiles. ¿Podría proporcionarme su número de orden para gestionarlo?",
            "category": "reembolsos",
            "priority": 2
        },
        {
            "question": "Mi reembolso aún no ha llegado",
            "answer": "Entiendo su preocupación. Los reembolsos pueden tardar hasta 7 días hábiles en reflejarse. ¿Desea que verifique el estado del suyo?",
            "category": "reembolsos",
            "priority": 2
        },
        {
            "question": "¿Cuáles son sus métodos de pago disponibles?",
            "answer": "Aceptamos pagos con tarjetas de crédito, débito, PayPal y transferencias bancarias. ¿Le gustaría ayuda para procesar su pago?",
            "category": "opciones_pago",
            "priority": 3
        },
        {
            "question": "Hice un pedido, pero no aparece en mi cuenta",
            "answer": "Es posible que haya realizado la compra como invitado. ¿Podría proporcionar su número de orden o el correo electrónico utilizado en la compra?",
            "category": "confirmacion_pedido",
            "priority": 1
        },
        {
            "question": "Quiero recibir una notificación cuando mi pedido sea enviado",
            "answer": "Puede activar las notificaciones en su cuenta. Si lo desea, también puedo verificar su número de orden para asegurarme de que reciba una alerta.",
            "category": "seguimiento_pedido",
            "priority": 3
        },
        {
            "question": "¿Cómo cambio la dirección de facturación?",
            "answer": "Puede actualizar su dirección de facturación desde su cuenta. Si su pedido ya ha sido procesado, puede que no sea posible cambiarla. ¿Le gustaría que verifique su orden?",
            "category": "problemas_facturacion",
            "priority": 2
        },
        {
            "question": "¿Puedo cambiar mi pedido después de haberlo pagado?",
            "answer": "Si su pedido aún no ha sido procesado, es posible modificarlo. ¿Podría proporcionarme su número de orden para verificar?",
            "category": "cambios_pedido",
            "priority": 2
        },
        {
            "question": "No recibí mi reembolso completo",
            "answer": "Algunos reembolsos pueden estar sujetos a tarifas bancarias o costos de procesamiento. ¿Podría indicarme su número de orden para revisar los detalles?",
            "category": "reembolsos",
            "priority": 2
        },
        {
            "question": "Quiero pagar en efectivo, ¿es posible?",
            "answer": "Actualmente, solo aceptamos pagos en línea. Sin embargo, puede usar métodos de pago digitales como transferencias bancarias o PayPal.",
            "category": "opciones_pago",
            "priority": 3
        }
        
    ],
    "use_gpu": True
}

# Test queries in Spanish
test_queries = [
    {
        "text": "Mi orden es ORD123456",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "Mi correo es usuario@ejemplo.com",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "¿Cuál es el estado de mi orden ORD123456?",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "¿Dónde está mi pedido?",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "Necesito mi factura",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "¿Puedo cambiar mi dirección de entrega?",
        "max_length": 50,
        "use_gpu": True
    }
]

order_related_cases = [
    {
        "question": f"Estado de la orden ORD000001",
        "answer": "He encontrado su orden ORD000001. [Detalles específicos serán insertados dinámicamente]",
        "category": "seguimiento_detallado",
        "priority": 1
    },
    {
        "question": "Mi orden fue cancelada",
        "answer": "Lamento escuchar eso. ¿Podría proporcionarme el número de orden para verificar el motivo de la cancelación y las opciones disponibles?",
        "category": "problemas_pedido",
        "priority": 1
    },
    {
        "question": "Mi pago fue declinado",
        "answer": "Entiendo su preocupación. ¿Podría proporcionarme el número de orden? Verificaré el motivo del rechazo y le ayudaré a procesarlo nuevamente.",
        "category": "problemas_pago",
        "priority": 1
    }
]

# Agregar los nuevos casos al test_cases existente
test_cases["cases"].extend(order_related_cases)

# Agregar nuevas consultas de prueba
additional_queries = [
    {
        "text": "¿Por qué fue cancelada mi orden ORD000001?",
        "max_length": 50,
        "use_gpu": True
    },
    {
        "text": "Mi pago fue rechazado en la orden ORD000002",
        "max_length": 50,
        "use_gpu": True
    }
]

test_queries.extend(additional_queries)
# Train the system
print("Entrenando el sistema...")
response = requests.post(
    "http://localhost:8002/train-support",
    json=test_cases
)
print("Respuesta del entrenamiento:", response.json())

# Test multiple queries
print("\nProbando casos similares para múltiples consultas:")
for query in test_queries:
    print(f"\nConsulta: {query['text']}")
    response = requests.post(
        "http://localhost:8002/get-similar-cases",
        json=query
    )
    print("Casos similares:", json.dumps(response.json(), indent=2, ensure_ascii=False))