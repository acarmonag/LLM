from datetime import datetime, timedelta
import random

class OrderDatabase:
    def __init__(self):
        self.orders = self._generate_orders()
        
    def _generate_orders(self):
        # Estados posibles de las órdenes
        statuses = {
            "pending": "Pendiente",
            "processing": "Procesando",
            "shipped": "Enviado",
            "delivered": "Entregado",
            "cancelled": "Cancelado",
            "declined": "Declinado",
            "refunded": "Reembolsado"
        }
        
        # Razones de cancelación/declinación
        decline_reasons = [
            "Pago rechazado por el banco",
            "Fondos insuficientes",
            "Problemas con la tarjeta",
            "Dirección de facturación incorrecta",
            "Sospecha de fraude"
        ]
        
        # Productos simulados
        products = [
            {"name": "Smartphone XYZ", "price": 799.99},
            {"name": "Laptop ABC", "price": 1299.99},
            {"name": "Auriculares Pro", "price": 199.99},
            {"name": "Tablet Ultra", "price": 499.99},
            {"name": "Smartwatch Plus", "price": 299.99}
        ]
        
        # Generar órdenes simuladas
        orders = {}
        current_date = datetime.now()
        
        # Generar 100 órdenes simuladas
        for i in range(1, 101):
            order_id = f"ORD{str(i).zfill(6)}"
            status = random.choice(list(statuses.keys()))
            order_date = current_date - timedelta(days=random.randint(0, 30))
            
            # Seleccionar productos aleatorios para la orden
            order_products = random.sample(products, random.randint(1, 3))
            total = sum(product["price"] for product in order_products)
            
            order = {
                "order_id": order_id,
                "customer_email": f"usuario{i}@ejemplo.com",
                "status": statuses[status],
                "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "products": order_products,
                "total": total,
                "payment_method": random.choice(["Credit Card", "PayPal", "Bank Transfer"]),
                "shipping_address": f"Calle {random.randint(1, 100)}, Ciudad",
                "tracking_number": f"TRACK{str(i).zfill(8)}" if status in ["shipped", "delivered"] else None
            }
            
            # Agregar información específica según el estado
            if status in ["cancelled", "declined"]:
                order["decline_reason"] = random.choice(decline_reasons)
                order["decline_date"] = (order_date + timedelta(hours=random.randint(1, 24))).strftime("%Y-%m-%d %H:%M:%S")
            
            if status == "refunded":
                order["refund_date"] = (order_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S")
                order["refund_amount"] = total
            
            if status in ["shipped", "delivered"]:
                order["shipping_date"] = (order_date + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d %H:%M:%S")
                if status == "delivered":
                    order["delivery_date"] = (datetime.strptime(order["shipping_date"], "%Y-%m-%d %H:%M:%S") + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S")
            
            orders[order_id] = order
        
        return orders
    
    def get_order(self, order_id):
        """Obtiene una orden por su ID."""
        return self.orders.get(order_id)
    
    def get_customer_orders(self, email):
        """Obtiene todas las órdenes de un cliente por email."""
        return [order for order in self.orders.values() if order["customer_email"] == email]
    
    def get_orders_by_status(self, status):
        """Obtiene todas las órdenes con un estado específico."""
        return [order for order in self.orders.values() if order["status"] == status]
    
    def get_order_status(self, order_id):
        """Obtiene el estado de una orden específica."""
        order = self.get_order(order_id)
        if order:
            return {
                "status": order["status"],
                "details": self._get_status_details(order)
            }
        return None
    
    def _get_status_details(self, order):
        """Obtiene detalles adicionales según el estado de la orden."""
        details = {
            "order_date": order["order_date"],
            "total": order["total"]
        }
        
        if order["status"] in ["Enviado", "Entregado"]:
            details["tracking_number"] = order["tracking_number"]
            details["shipping_date"] = order["shipping_date"]
            if order["status"] == "Entregado":
                details["delivery_date"] = order["delivery_date"]
        
        if order["status"] in ["Cancelado", "Declinado"]:
            details["decline_reason"] = order["decline_reason"]
            details["decline_date"] = order["decline_date"]
        
        if order["status"] == "Reembolsado":
            details["refund_date"] = order["refund_date"]
            details["refund_amount"] = order["refund_amount"]
            
        return details