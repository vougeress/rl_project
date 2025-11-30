import { Product } from '../../types';
import { Star } from 'lucide-react';
import { ImageWithFallback } from '../figma/ImageWithFallback';
import './ProductCard.css';

interface ProductCardProps {
  product: Product;
  onClick: () => void;
}

export function ProductCard({ product, onClick }: ProductCardProps) {
  return (
    <div className="product-card" onClick={onClick}>
      <div className="product-image-wrapper">
        <ImageWithFallback
          src={product.image_url || ''}
          alt={product.product_name}
          className="product-image"
        />
      </div>
      
      <div className="product-info">
        <div className="product-category">{product.category_name}</div>
        <h3 className="product-name">{product.product_name}</h3>
        
        <div className="product-footer">
          <span className="product-price">${product.price.toFixed(2)}</span>
          <div className="product-quality">
            <Star className="quality-star" size={16} fill="currentColor" />
            <span>{product.quality.toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
