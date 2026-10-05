package openLLM.cli;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MainTest {
    @Test void mainHasAGreeting() {
        assertNotNull(new Main().getGreeting());
    }
}