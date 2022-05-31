import React, { cloneElement, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './Tooltip.module.css';
import {
  useFloating,
  useInteractions,
  useHover,
  offset,
  flip,
  autoUpdate,
} from '@floating-ui/react-dom-interactions';

const Tooltip = ({ label, children }) => {
  const [isOpen, setIsOpen] = useState(false);
  const { x, y, reference, floating, strategy, context } = useFloating({
    open: isOpen,
    onOpenChange: setIsOpen,
    middleware: [offset(5), flip()],
    whileElementsMounted: autoUpdate,
  });

  const { getReferenceProps, getFloatingProps } = useInteractions([
    useHover(context, {
      delay: { open: 500, close: 10 },
    }),
  ]);

  return (
    <>
      {cloneElement(
        children,
        getReferenceProps({ ref: reference, ...children.props })
      )}
      {isOpen &&
        createPortal(
          <div
            {...getFloatingProps({
              ref: floating,
              className: styles.root,
              style: {
                position: strategy,
                top: y ?? '',
                left: x ?? '',
              },
            })}
          >
            {label}
          </div>,
          document.body
        )}
    </>
  );
};

export default Tooltip;
